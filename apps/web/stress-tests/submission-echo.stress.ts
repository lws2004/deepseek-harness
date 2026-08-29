/**
 * Opt-in browser stress reproduction for delayed submission echo in long
 * sessions. The design contract (packages/client/ui-conversation/src/client/
 * input/service.ts) paints the echo bubble on the send keystroke's own frame,
 * before serialization or transport; user reports say the bubble itself lands
 * late in long histories. The test seeds a 500-turn session, loads it fully,
 * and measures wall-clock time from the Send click to the
 * `[data-submission-echo]` node entering the DOM, against a short-session
 * baseline.
 */
import { mkdtemp, readdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { performance } from 'node:perf_hooks'
import type { Browser, Page } from 'playwright'
import { chromium } from 'playwright'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import type { StreamChunk } from '@deepseek-ai/dsh-llm'
import {
  ToolCallId,
  createAssistantMessage,
  createToolResultMessage,
  createUserMessage,
} from '@deepseek-ai/dsh-llm'
import type { ReplayEntry, ReplayOverrideDoc } from '@deepseek-ai/dsh-llm-replay'
import { Session, SessionId, SESSION_FORMAT_VERSION } from '@deepseek-ai/dsh-session'
// Carries the session/title event declaration into the fixture builder.
import type {} from '@deepseek-ai/dsh-session-title'
import {
  launchWebScaffold,
  seedSession,
  watchConsole,
  type WebScaffold,
} from '../tests/scaffold.ts'
import { connectFreshWorkspace, newEnglishPage, saveFailureShot } from '../tests/support.ts'

const LONG_SESSION_ID = 'echo-long-history'
const LONG_HISTORY_TURNS = 500
const TOOL_TURN_INTERVAL = 10
const TOOLS_PER_TOOL_TURN = 10
const PERF_REPLAY_CONTEXT_WINDOW = 10_000_000
/** Prompt sent once per scenario; the replay override answers each in order. */
const ECHO_PROMPT = (scenario: string): string =>
  `ECHO_STRESS_${scenario}: reply with one short sentence.`
const ECHO_REPLY = (scenario: string): string =>
  `Echo stress reply for ${scenario}.`

interface StressWorld {
  readonly scaffold: WebScaffold
  readonly page: Page
  readonly replayDir?: string
  setupMs: number
}

function text(value: string): { type: 'text'; text: string }[] {
  return [{ type: 'text', text: value }]
}

function appendTitle(session: Session, title: string, messageSeq: number): void {
  session.append('session/title', {
    title,
    messageSeqs: [messageSeq],
    source: { kind: 'fallback' },
  })
}

function appendRequestHeader(session: Session, turn: number, step: number): void {
  session.append('request/header', {
    header: {
      config: { provider: 'deepseek-official', model: 'deepseek-v4-flash' },
      system: `Synthetic echo-stress system prompt for turn ${String(turn)}, step ${String(step)}.`,
    },
    reason: turn === 1 && step === 1 ? 'initial' : 'change',
  })
}

function appendAssistant(session: Session, turn: number, step: number, body: string): void {
  session.append('assistant/message', {
    turn,
    step,
    message: createAssistantMessage({
      content: text(body),
      source: { provider: 'deepseek-official', model: 'deepseek-v4-flash' },
    }),
    usage: {
      inputTokens: 4_000 + turn * 10,
      outputTokens: 200 + step * 20,
      cacheReadTokens: turn % 2 === 0 ? 2_000 : 0,
    },
  }, { surfaceOp: 'append' })
}

function appendToolStep(session: Session, turn: number, step: number, toolCount: number): void {
  const calls = Array.from({ length: toolCount }, (_, index) => ({
    callId: ToolCallId(`echo-call-${String(turn)}-${String(index)}`),
    index,
    args: JSON.stringify({ turn, index, payload: 'x'.repeat(120) }),
  }))

  session.append('assistant/message', {
    turn,
    step,
    message: createAssistantMessage({
      content: [
        {
          type: 'reasoning',
          text: `Dispatching ${String(toolCount)} synthetic tools for turn ${String(turn)}.`,
        },
        ...calls.map(({ callId, args }) => ({
          type: 'tool-call' as const,
          id: callId,
          name: 'synthetic_tool',
          arguments: args,
        })),
      ],
      source: { provider: 'deepseek-official', model: 'deepseek-v4-flash' },
    }),
    usage: {
      inputTokens: 6_000 + turn * 10,
      outputTokens: 500,
      cacheReadTokens: 3_000,
      reasoningTokens: 50,
    },
  }, { surfaceOp: 'append' })

  const callEvents = calls.map(({ callId, args }) =>
    session.append('tool/call', {
      turn,
      step,
      callId,
      name: 'synthetic_tool',
      arguments: args,
    }))

  for (const [index, call] of calls.entries()) {
    const source = callEvents[index]
    if (source === undefined) throw new Error(`missing synthetic tool call ${String(index)}`)
    session.append('tool/result', {
      turn,
      step,
      message: createToolResultMessage({
        callId: call.callId,
        content: text(
          `synthetic result turn=${String(turn)} index=${String(call.index)} ${'r'.repeat(400)}`,
        ),
        isError: false,
      }),
    }, { surfaceOp: 'append', sourceEventSeqs: [source.seq] })
  }
}

function fencedCode(turn: number): string {
  if (turn % 25 !== 0) return ''
  const lines = Array.from(
    { length: 80 },
    (_, index) => `const value_${String(index)} = ${String(turn + index)}`,
  )
  return `\n\n\`\`\`ts\n${lines.join('\n')}\n\`\`\``
}

function fixtureLog(session: Session): string {
  const header = {
    type: 'session',
    version: SESSION_FORMAT_VERSION,
    id: '{{sessionId}}',
    createdAt: Date.now() - 60_000,
    cwd: '{{cwd}}',
  }
  return [
    JSON.stringify(header),
    ...session.events.map(event => JSON.stringify(event)),
    '',
  ].join('\n')
}

function longHistoryFixture(): string {
  const session = Session.create(SessionId(LONG_SESSION_ID))
  for (let turn = 1; turn <= LONG_HISTORY_TURNS; turn += 1) {
    session.append('turn/start', { turn })
    const user = session.append('user/message', createUserMessage({
      content: text(
        `ECHO_STRESS_HISTORY turn ${String(turn)}: analyze payload ${'u'.repeat(200)}`,
      ),
      source: { kind: 'user' },
    }), { surfaceOp: 'append' })
    if (turn === 1) appendTitle(session, 'ECHO_STRESS_HISTORY long session', user.seq)

    session.append('step/start', { turn, step: 1 })
    appendRequestHeader(session, turn, 1)
    if (turn % TOOL_TURN_INTERVAL === 0) {
      appendToolStep(session, turn, 1, TOOLS_PER_TOOL_TURN)
      session.append('step/end', { turn, step: 1 })
      session.append('step/start', { turn, step: 2 })
      appendRequestHeader(session, turn, 2)
      appendAssistant(
        session,
        turn,
        2,
        `All synthetic tools completed for turn ${String(turn)}. `
          + `${'z'.repeat(320)}${fencedCode(turn)}`,
      )
      session.append('step/end', { turn, step: 2 })
    } else {
      appendAssistant(
        session,
        turn,
        1,
        `Synthetic assistant response for turn ${String(turn)}. `
          + `${'a'.repeat(320)}${fencedCode(turn)}`,
      )
      session.append('step/end', { turn, step: 1 })
    }
    session.append('turn/end', { turn, reason: { kind: 'completed' } })
  }
  return fixtureLog(session)
}

function textStream(responseText: string, inputTokens: number): StreamChunk[] {
  return [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'text-delta', index: 0, text: responseText },
    {
      type: 'block-end',
      index: 0,
      block: { type: 'text', text: responseText },
    },
    {
      type: 'usage',
      usage: { inputTokens, outputTokens: Math.ceil(responseText.length / 4) },
    },
    { type: 'finish', reason: { kind: 'stop' } },
  ]
}

function echoReplayOverride(scenarios: readonly string[]): ReplayOverrideDoc {
  const entries: ReplayEntry[] = scenarios.map(scenario => ({
    kind: 'chunks',
    chunks: textStream(ECHO_REPLY(scenario), 4_000),
  }))
  return entries
}

async function launchStressWorld(
  browser: Browser,
  replayScenarios: readonly string[],
): Promise<StressWorld> {
  const setupStarted = performance.now()
  let scaffold: WebScaffold | undefined
  let page: Page | undefined
  let replayDir: string | undefined
  try {
    replayDir = await mkdtemp(join(tmpdir(), 'dsh-web-echo-stress-'))
    const replayOverride = join(replayDir, 'replay.override.json')
    await writeFile(replayOverride, JSON.stringify(echoReplayOverride(replayScenarios)))
    scaffold = await launchWebScaffold({
      replayFixture: join(replayDir, 'override-only.jsonl'),
      replayOverride,
      replayContextWindow: PERF_REPLAY_CONTEXT_WINDOW,
    })
    await seedSession(scaffold, longHistoryFixture(), LONG_SESSION_ID)
    page = await newEnglishPage(browser)
    const world: StressWorld = {
      scaffold,
      page,
      replayDir,
      setupMs: performance.now() - setupStarted,
    }
    return world
  } catch (error) {
    await page?.close().catch(() => {})
    await scaffold?.close().catch(() => {})
    if (replayDir !== undefined) await rm(replayDir, { recursive: true, force: true }).catch(() => {})
    throw error
  }
}

async function closeStressWorld(world: StressWorld): Promise<void> {
  await world.page.close().catch(() => {})
  await world.scaffold.close().catch(() => {})
  if (world.replayDir !== undefined) {
    await rm(world.replayDir, { recursive: true, force: true }).catch(() => {})
  }
}

/** Install a capture-phase observer that timestamps the first echo bubble. */
async function armEchoObserver(page: Page): Promise<void> {
  await page.evaluate(() => {
    const w = globalThis as typeof globalThis & { __echoAt?: number }
    delete w.__echoAt
    const observer = new MutationObserver(() => {
      if (document.querySelector('[data-submission-echo]') !== null) {
        w.__echoAt = globalThis.performance.now()
        observer.disconnect()
      }
    })
    observer.observe(document.body, { childList: true, subtree: true })
  })
}

async function readEchoLatencyMs(page: Page): Promise<number | undefined> {
  return page.evaluate(() => (globalThis as typeof globalThis & { __echoAt?: number }).__echoAt)
}

/**
 * Open one session by its sidebar search sentinel and measure the Send →
 * echo-bubble latency for one prompt. Loads every history page first when
 * `loadAllHistory` is set, so the measurement reflects the fully expanded
 * transcript.
 */
async function measureEchoLatency(
  world: StressWorld,
  options: {
    readonly scenario: string
    readonly sentinel: string
    readonly loadAllHistory: boolean
    readonly expectedTurnsAfterLoad: number
    /** Stay in the already-open session (blank baseline) instead of navigating. */
    readonly skipSidebarNavigation?: boolean
  },
): Promise<{ echoMs: number; durableMs: number }> {
  const { page, scaffold } = world
  if (options.skipSidebarNavigation !== true) {
    // Seeded sessions are outside every workspace's account (the sidebar tree
    // groups by workspace membership and never lists them), while the content
    // index covers the seeded log deterministically — the same search-
    // navigation route the other seeded-session lanes use. Search is a
    // collapsed header action; expand it so the input is actionable.
    const searchButton = page.getByRole('button', { name: 'Search sessions' })
    if (await searchButton.getAttribute('aria-expanded') !== 'true') await searchButton.click()
    const search = page.getByPlaceholder('Search sessions', { exact: false })
    await search.fill(options.sentinel)
    const result = page.getByRole('tree', { name: 'Search results' }).getByRole('treeitem')
    await expect.poll(() => result.count(), { timeout: 30_000 }).toBeGreaterThanOrEqual(1)
    const hitTexts = await result.allTextContents()
    console.info(`ECHO_STRESS_SEARCH_HITS ${JSON.stringify(hitTexts.slice(0, 5))}`)
    await result.click()
    await page.getByRole('tab', { name: 'Chat', exact: true }).waitFor({ timeout: 30_000 })
  }

  if (options.loadAllHistory) {
    let turns = await conversationTurns(page)
    while (turns < options.expectedTurnsAfterLoad) {
      await page.getByRole('button', { name: 'Load earlier', exact: true }).click()
      await expect.poll(() => conversationTurns(page), { timeout: 30_000 })
        .toBeGreaterThan(turns)
      turns = await conversationTurns(page)
    }
  }

  const composer = page.locator('[data-composer-input][contenteditable="true"]').last()
  await composer.waitFor({ timeout: 15_000 })
  const prompt = ECHO_PROMPT(options.scenario)
  await composer.fill(prompt)
  await expect.poll(() => composer.textContent()).toBe(prompt)

  await armEchoObserver(page)
  const settled = scaffold.whenTurnSettled(60_000)
  const clickAt = await page.evaluate(() => globalThis.performance.now())
  await page.getByRole('button', { name: 'Send message', exact: true }).click()
  await expect
    .poll(() => readEchoLatencyMs(page), { timeout: 15_000 })
    .toBeDefined()
  const echoAt = (await readEchoLatencyMs(page)) ?? 0
  await page.getByText(prompt, { exact: false }).last().waitFor({ timeout: 15_000 })
  const durableMs = await page.evaluate(
    (start: number) => globalThis.performance.now() - start,
    clickAt,
  )
  await settled
  await expect.poll(() => page.locator('[data-streaming="true"]').count(), { timeout: 15_000 }).toBe(0)
  return { echoMs: echoAt - clickAt, durableMs }
}

/** Loaded-window turn count: one mounted turn-tail row per settled turn. */
async function conversationTurns(page: Page): Promise<number> {
  return page.locator('[data-chat-flow-key^="9:turn-tail"]').count()
}

let browser: Browser | undefined

beforeAll(async () => {
  browser = await chromium.launch({ headless: process.env.DSH_WEB_STRESS_HEADFUL !== '1' })
})

afterAll(async () => {
  await browser?.close()
})

describe('submission echo latency vs history length', () => {
  it('reports Send → echo-bubble latency for short and long sessions', async () => {
    if (browser === undefined) throw new Error('browser unavailable')
    const world = await launchStressWorld(browser, ['short', 'long-default', 'long-expanded'])
    const tripwire = watchConsole(world.page)
    let testFailure: unknown
    try {
      const { page, scaffold } = world
      await page.goto(scaffold.authenticatedUrl, { waitUntil: 'load' })
      await page.waitForSelector('[class*="frame"]', { timeout: 30_000 })

      // Blank-session baseline: connect a fresh workspace and send from the
      // brand-new empty session.
      await connectFreshWorkspace(page, scaffold.workspaceCwd, 'echo-stress-baseline')
      const short = await measureEchoLatency(world, {
        scenario: 'short',
        sentinel: 'ECHO_STRESS_HISTORY turn 1',
        loadAllHistory: false,
        expectedTurnsAfterLoad: 0,
        skipSidebarNavigation: true,
      })
      const longDefault = await measureEchoLatency(world, {
        scenario: 'long-default',
        sentinel: 'ECHO_STRESS_HISTORY turn 1',
        loadAllHistory: false,
        expectedTurnsAfterLoad: 0,
      })
      const longExpanded = await measureEchoLatency(world, {
        scenario: 'long-expanded',
        sentinel: 'ECHO_STRESS_HISTORY turn 1',
        loadAllHistory: true,
        expectedTurnsAfterLoad: LONG_HISTORY_TURNS,
      })

      const report = {
        setupMs: Math.round(world.setupMs),
        shortEchoMs: Math.round(short.echoMs),
        longDefaultEchoMs: Math.round(longDefault.echoMs),
        longExpandedEchoMs: Math.round(longExpanded.echoMs),
        shortDurableMs: Math.round(short.durableMs),
        longDefaultDurableMs: Math.round(longDefault.durableMs),
        longExpandedDurableMs: Math.round(longExpanded.durableMs),
      }
      console.info(`ECHO_STRESS_RESULT ${JSON.stringify(report)}`)
      expect(await world.page.getByRole('tab', { name: 'Chat', exact: true }).isVisible()).toBe(true)
      expect(tripwire.pageErrors).toEqual([])
    } catch (error) {
      testFailure = error
      throw error
    } finally {
      if (testFailure !== undefined) {
        const bodyText = await world.page.evaluate(() =>
          Array.from(document.querySelectorAll('[role=treeitem]'))
            .map(element => element.textContent)
            .join(' | ')
            .slice(0, 1500),
        )
          .catch(error => `body dump failed: ${String(error)}`)
        console.info(`ECHO_STRESS_PAGE_DUMP ${JSON.stringify(bodyText)}`)
        const seeded = await readdir(world.scaffold.persistenceRoot, { recursive: true })
          .catch(error => [String(error)])
        console.info(`ECHO_STRESS_SEED_FILES ${JSON.stringify(seeded.slice(0, 30))}`)
        const hostIds = await world.scaffold.ctx.sessionQuery.listSessions()
          .then(records => records.map(record => String(record.header.id)))
          .catch(error => [`list failed: ${String(error)}`])
        console.info(`ECHO_STRESS_HOST_LIST ${JSON.stringify(hostIds)}`)
        await saveFailureShot(world.page, 'web-stress-submission-echo')
      }
      await closeStressWorld(world)
    }
  })
})
