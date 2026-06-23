#!/usr/bin/env tsx
/**
 * @license
 * Copyright 2025 Vybestack LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { spawn } from 'node:child_process';
import { constants as fsConstants } from 'node:fs';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '..');
const defaultProvider = 'gemini';
const defaultModel = 'gemini-2.5-pro';
const defaultLlxprtBin = './bundle/llxprt.js';

interface CliOptions {
  inputPath: string;
  context?: string;
  contextFile?: string;
  outputDir?: string;
  provider: string;
  model: string;
  llxprtBin: string;
  audioBitrate: string;
  opusBitrate: string;
  language?: string;
  speakerHint?: string;
  skipGemini: boolean;
}

interface ProcessResult {
  stdout: string;
  stderr: string;
}

interface AudioProbe {
  format?: {
    duration?: string;
    format_name?: string;
    size?: string;
  };
  streams?: Array<{
    codec_name?: string;
    codec_type?: string;
    channels?: number;
    sample_rate?: string;
  }>;
}

function usage(): string {
  return `Usage:
  npm run transcribe-audio -- <audio-file> [options]

Options:
  --context <text>          Extra transcription context.
  --context-file <path>     Read extra transcription context from a file.
  --output-dir <path>       Output bundle directory. Defaults to .llxprt-audio/<run-id>.
  --provider <name>         LLxprt provider. Defaults to ${defaultProvider}.
  --model <name>            Gemini multimodal or STT model. Defaults to ${defaultModel}.
  --llxprt-bin <command>    LLxprt command. Defaults to ${defaultLlxprtBin}.
  --audio-bitrate <rate>    AAC/M4A bitrate. Defaults to 96k.
  --opus-bitrate <rate>     Opus bitrate. Defaults to 48k.
  --language <name>         Optional expected language.
  --speakers <hint>         Optional speaker count or names.
  --skip-gemini             Convert/package audio without calling LLxprt.
  --help                    Show this help.

Examples:
  npm run transcribe-audio -- ~/recordings/interview.m4a --context "Property auction prep call"
  npm run transcribe-audio -- ./audio.wav --model gemini-3-flash-preview --language English
  npm run transcribe-audio -- ./audio.mp3 --skip-gemini --output-dir ./tmp/audio-smoke`;
}

function requireValue(args: string[], index: number, flag: string): string {
  const value = args[index + 1];
  if (value === undefined || value.startsWith('--')) {
    throw new Error(`${flag} requires a value.`);
  }
  return value;
}

function parseArgs(args: string[]): CliOptions {
  if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    process.stdout.write(`${usage()}\n`);
    process.exit(0);
  }

  let inputPath: string | undefined;
  const options: Omit<CliOptions, 'inputPath'> = {
    provider: defaultProvider,
    model: defaultModel,
    llxprtBin: defaultLlxprtBin,
    audioBitrate: '96k',
    opusBitrate: '48k',
    skipGemini: false,
  };

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (!arg.startsWith('--')) {
      if (inputPath !== undefined) {
        throw new Error(`Unexpected positional argument: ${arg}`);
      }
      inputPath = arg;
      continue;
    }

    switch (arg) {
      case '--context':
        options.context = requireValue(args, index, arg);
        index += 1;
        break;
      case '--context-file':
        options.contextFile = requireValue(args, index, arg);
        index += 1;
        break;
      case '--output-dir':
        options.outputDir = requireValue(args, index, arg);
        index += 1;
        break;
      case '--provider':
        options.provider = requireValue(args, index, arg);
        index += 1;
        break;
      case '--model':
        options.model = requireValue(args, index, arg);
        index += 1;
        break;
      case '--llxprt-bin':
        options.llxprtBin = requireValue(args, index, arg);
        index += 1;
        break;
      case '--audio-bitrate':
        options.audioBitrate = requireValue(args, index, arg);
        index += 1;
        break;
      case '--opus-bitrate':
        options.opusBitrate = requireValue(args, index, arg);
        index += 1;
        break;
      case '--language':
        options.language = requireValue(args, index, arg);
        index += 1;
        break;
      case '--speakers':
        options.speakerHint = requireValue(args, index, arg);
        index += 1;
        break;
      case '--skip-gemini':
        options.skipGemini = true;
        break;
      default:
        throw new Error(`Unknown option: ${arg}`);
    }
  }

  if (inputPath === undefined) {
    throw new Error('Missing required audio file path.');
  }

  return { inputPath, ...options };
}

async function pathExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

function isInside(parent: string, child: string): boolean {
  const relative = path.relative(parent, child);
  return (
    relative === '' ||
    (!relative.startsWith('..') && !path.isAbsolute(relative))
  );
}

function slugify(value: string): string {
  const slug = value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return slug.length > 0 ? slug.slice(0, 64) : 'audio';
}

function timestamp(): string {
  return new Date().toISOString().replace(/[:.]/g, '-');
}

async function run(
  command: string,
  args: string[],
  options: { cwd?: string; env?: NodeJS.ProcessEnv; label?: string } = {},
): Promise<ProcessResult> {
  return await new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: options.cwd,
      env: options.env,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    const stdoutChunks: Buffer[] = [];
    const stderrChunks: Buffer[] = [];

    child.stdout.on('data', (chunk: Buffer) => stdoutChunks.push(chunk));
    child.stderr.on('data', (chunk: Buffer) => stderrChunks.push(chunk));
    child.on('error', reject);
    child.on('close', (code) => {
      const stdout = Buffer.concat(stdoutChunks).toString('utf8');
      const stderr = Buffer.concat(stderrChunks).toString('utf8');
      if (code === 0) {
        resolve({ stdout, stderr });
        return;
      }
      const label = options.label ?? command;
      reject(
        new Error(
          `${label} exited with code ${String(code)}${stderr.length > 0 ? `:\n${stderr}` : ''}`,
        ),
      );
    });
  });
}

async function requireCommand(command: string): Promise<void> {
  if (command.includes('/') || command.includes('\\')) {
    const commandPath = path.isAbsolute(command)
      ? command
      : path.resolve(repoRoot, command);
    try {
      await fs.access(commandPath, fsConstants.X_OK);
      return;
    } catch {
      const hint =
        command === defaultLlxprtBin
          ? ' Run `npm run bundle` from the repo root first.'
          : '';
      throw new Error(`Command is not executable: ${commandPath}.${hint}`);
    }
  }

  const lookupCommand = process.platform === 'win32' ? 'where' : 'sh';
  const lookupArgs =
    process.platform === 'win32'
      ? [command]
      : ['-lc', `command -v ${command} >/dev/null`];

  await run(lookupCommand, lookupArgs, { label: `${command} lookup` });
}

async function probeAudio(inputPath: string): Promise<AudioProbe> {
  const { stdout } = await run(
    'ffprobe',
    ['-v', 'error', '-show_format', '-show_streams', '-of', 'json', inputPath],
    { label: 'ffprobe' },
  );
  return JSON.parse(stdout) as AudioProbe;
}

async function convertForGemini(
  inputPath: string,
  outputPath: string,
): Promise<void> {
  await run(
    'ffmpeg',
    [
      '-y',
      '-hide_banner',
      '-loglevel',
      'error',
      '-i',
      inputPath,
      '-vn',
      '-ac',
      '1',
      '-ar',
      '16000',
      '-c:a',
      'pcm_s16le',
      outputPath,
    ],
    { label: 'ffmpeg wav conversion' },
  );
}

async function exportCompressedAudio(
  inputPath: string,
  outputDir: string,
  audioBitrate: string,
  opusBitrate: string,
): Promise<{ m4aPath: string; opusPath: string }> {
  const m4aPath = path.join(outputDir, 'audio.m4a');
  const opusPath = path.join(outputDir, 'audio.opus');

  await run(
    'ffmpeg',
    [
      '-y',
      '-hide_banner',
      '-loglevel',
      'error',
      '-i',
      inputPath,
      '-vn',
      '-c:a',
      'aac',
      '-b:a',
      audioBitrate,
      '-movflags',
      '+faststart',
      m4aPath,
    ],
    { label: 'ffmpeg m4a export' },
  );

  await run(
    'ffmpeg',
    [
      '-y',
      '-hide_banner',
      '-loglevel',
      'error',
      '-i',
      inputPath,
      '-vn',
      '-c:a',
      'libopus',
      '-b:a',
      opusBitrate,
      '-vbr',
      'on',
      opusPath,
    ],
    { label: 'ffmpeg opus export' },
  );

  return { m4aPath, opusPath };
}

async function fileSize(filePath: string): Promise<number> {
  const stats = await fs.stat(filePath);
  return stats.size;
}

function formatSeconds(duration?: string): string | undefined {
  if (duration === undefined) {
    return undefined;
  }
  const seconds = Number(duration);
  if (!Number.isFinite(seconds)) {
    return duration;
  }
  const whole = Math.round(seconds);
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const secs = whole % 60;
  return [hours, minutes, secs]
    .map((part) => String(part).padStart(2, '0'))
    .join(':');
}

async function buildContext(options: CliOptions): Promise<string> {
  const parts: string[] = [];
  if (options.context !== undefined) {
    parts.push(options.context);
  }
  if (options.contextFile !== undefined) {
    const contextPath = path.resolve(options.contextFile);
    parts.push(await fs.readFile(contextPath, 'utf8'));
  }
  return parts.join('\n\n').trim();
}

function buildPrompt(params: {
  audioPath: string;
  context: string;
  language?: string;
  speakerHint?: string;
}): string {
  const lines = [
    'Transcribe the attached audio file.',
    '',
    'Return Markdown only with these sections:',
    '# Transcript',
    '## Summary',
    '## Speakers',
    '## Keywords',
    '## Segments',
    '',
    'Requirements:',
    '- Produce a timestamped diarized transcript.',
    '- Use this transcript line format: [HH:MM:SS - HH:MM:SS] Speaker N: text',
    '- Keep speaker labels stable across the file.',
    '- Preserve meaningful pauses, uncertainty, and inaudible moments with concise bracketed notes.',
    '- In ## Segments, include a Markdown table with start, end, speaker, and text columns.',
    '- Do not summarize instead of transcribing. Include the full transcript.',
  ];

  if (params.language !== undefined) {
    lines.push(`- Expected language: ${params.language}`);
  }
  if (params.speakerHint !== undefined) {
    lines.push(`- Speaker hint: ${params.speakerHint}`);
  }
  if (params.context.length > 0) {
    lines.push('', 'Context:', params.context);
  }

  lines.push('', `Audio: @${params.audioPath}`);
  return lines.join('\n');
}

function llxprtArgs(options: CliOptions, prompt: string): string[] {
  const args =
    options.llxprtBin === 'npx' ? ['--yes', '@vybestack/llxprt-code'] : [];

  args.push(
    '--provider',
    options.provider,
    '--model',
    options.model,
    '--yolo',
    '--output-format',
    'text',
    prompt,
  );

  return args;
}

async function writeMetadata(params: {
  outputDir: string;
  sourcePath: string;
  probe: AudioProbe;
  geminiInputPath: string;
  geminiInputSize: number;
  m4aPath: string;
  opusPath: string;
  options: CliOptions;
}): Promise<void> {
  const audioStream = params.probe.streams?.find(
    (stream) => stream.codec_type === 'audio',
  );
  const metadata = {
    createdAt: new Date().toISOString(),
    sourcePath: params.sourcePath,
    source: {
      format: params.probe.format?.format_name,
      duration: params.probe.format?.duration,
      durationHms: formatSeconds(params.probe.format?.duration),
      size: params.probe.format?.size,
      audioCodec: audioStream?.codec_name,
      sampleRate: audioStream?.sample_rate,
      channels: audioStream?.channels,
    },
    geminiInput: {
      path: params.geminiInputPath,
      format: 'wav',
      codec: 'pcm_s16le',
      sampleRate: 16000,
      channels: 1,
      size: params.geminiInputSize,
    },
    publishingAudio: {
      m4a: params.m4aPath,
      opus: params.opusPath,
    },
    llxprt: {
      provider: params.options.provider,
      model: params.options.model,
      command:
        params.options.llxprtBin === 'npx'
          ? 'npx --yes @vybestack/llxprt-code'
          : params.options.llxprtBin,
      skipped: params.options.skipGemini,
    },
  };
  await fs.writeFile(
    path.join(params.outputDir, 'metadata.json'),
    `${JSON.stringify(metadata, null, 2)}\n`,
  );
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const sourcePath = path.resolve(options.inputPath);
  if (!(await pathExists(sourcePath))) {
    throw new Error(`Audio file does not exist: ${sourcePath}`);
  }

  await requireCommand('ffmpeg');
  await requireCommand('ffprobe');
  if (!options.skipGemini && options.llxprtBin !== 'npx') {
    await requireCommand(options.llxprtBin);
  }

  const runId = `${slugify(path.basename(sourcePath, path.extname(sourcePath)))}-${timestamp()}`;
  const outputDir = path.resolve(
    options.outputDir ?? path.join(repoRoot, '.llxprt-audio', runId),
  );
  const workspaceOutput = isInside(repoRoot, outputDir);
  const stageDir = workspaceOutput
    ? outputDir
    : path.join(repoRoot, '.llxprt-audio', 'staging', runId);

  await fs.mkdir(outputDir, { recursive: true });
  await fs.mkdir(stageDir, { recursive: true });

  const probe = await probeAudio(sourcePath);
  const geminiInputPath = path.join(stageDir, 'gemini-input.wav');
  const promptPath = path.join(outputDir, 'prompt.md');
  const transcriptRawPath = path.join(outputDir, 'transcript.raw.txt');
  const transcriptMarkdownPath = path.join(outputDir, 'transcript.md');

  process.stdout.write(`Preparing audio bundle in ${outputDir}\n`);
  await convertForGemini(sourcePath, geminiInputPath);
  if (stageDir !== outputDir) {
    await fs.copyFile(
      geminiInputPath,
      path.join(outputDir, 'gemini-input.wav'),
    );
  }
  const { m4aPath, opusPath } = await exportCompressedAudio(
    sourcePath,
    outputDir,
    options.audioBitrate,
    options.opusBitrate,
  );
  const geminiInputSize = await fileSize(geminiInputPath);

  const context = await buildContext(options);
  const prompt = buildPrompt({
    audioPath: geminiInputPath,
    context,
    language: options.language,
    speakerHint: options.speakerHint,
  });
  await fs.writeFile(promptPath, `${prompt}\n`);
  await writeMetadata({
    outputDir,
    sourcePath,
    probe,
    geminiInputPath,
    geminiInputSize,
    m4aPath,
    opusPath,
    options,
  });

  if (options.skipGemini) {
    await fs.writeFile(
      transcriptRawPath,
      'Gemini transcription was skipped. Run again without --skip-gemini to generate transcript output.\n',
    );
    await fs.writeFile(
      transcriptMarkdownPath,
      '# Transcript\n\nGemini transcription was skipped. Run again without `--skip-gemini` to generate transcript output.\n',
    );
    process.stdout.write(
      `Wrote audio bundle without Gemini transcript: ${outputDir}\n`,
    );
    return;
  }

  process.stdout.write(`Calling LLxprt ${options.provider}/${options.model}\n`);
  const result = await run(options.llxprtBin, llxprtArgs(options, prompt), {
    cwd: repoRoot,
    env: process.env,
    label: 'llxprt transcription',
  });
  const transcript = result.stdout.trimEnd();
  await fs.writeFile(
    transcriptRawPath,
    `${result.stdout}${result.stderr.length > 0 ? `\n\n--- STDERR ---\n${result.stderr}` : ''}`,
  );
  await fs.writeFile(transcriptMarkdownPath, `${transcript}\n`);
  process.stdout.write(`Wrote transcript bundle: ${outputDir}\n`);
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`transcribe-audio failed: ${message}\n`);
  process.exitCode = 1;
});
