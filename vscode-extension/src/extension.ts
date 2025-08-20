import * as vscode from 'vscode';
import { ChildProcessWithoutNullStreams, spawn } from 'child_process';

let proc: ChildProcessWithoutNullStreams | null = null;
let output: vscode.OutputChannel | null = null;
let ready = false;
let nextId = 1;

function ensureOutput(): vscode.OutputChannel {
  if (!output) {
    output = vscode.window.createOutputChannel('D-LinOSS MCP');
  }
  return output;
}

function sendRpc(method: string, params: any = {}): void {
  if (!proc || !ready) {
    vscode.window.showWarningMessage('MCP session not ready. Run: D-LinOSS: Start MCP Session');
    return;
  }
  const id = nextId++;
  const msg = JSON.stringify({ jsonrpc: '2.0', id, method, params });
  proc.stdin.write(msg + '\n');
  ensureOutput().appendLine(`➡ ${msg}`);
}

export function activate(context: vscode.ExtensionContext) {
  const startCmd = vscode.commands.registerCommand('dlinoss.mcp.start', async () => {
    if (proc) {
      vscode.window.showInformationMessage('MCP session already running.');
      return;
    }
    const useFft = await vscode.window.showQuickPick(['yes','no'], { title: 'Enable FFT feature?', placeHolder: 'yes/no' });
    const args = ['run','-p','xtask','--','mcp-serve'];
    if (useFft === 'yes') {
      args.push('--fft');
    }
    ensureOutput().show(true);
    ensureOutput().appendLine(`[ext] Spawning: cargo ${args.join(' ')}`);
    proc = spawn('cargo', args, { cwd: vscode.workspace.workspaceFolders?.[0].uri.fsPath });
    proc.stdout.on('data', (data: Buffer) => {
      const text = data.toString();
      text.split(/\r?\n/).filter(l => l.length>0).forEach(line => {
        ensureOutput().appendLine(line);
        if (!ready && line.includes('"mcp_ready":true')) {
          ready = true;
          ensureOutput().appendLine('[ext] MCP ready, sending ping');
          sendRpc('dlinoss.ping');
        } else {
          // Try parse response lines
          try {
            const parsed = JSON.parse(line);
            if (parsed.result) {
              ensureOutput().appendLine(`✔ result: ${JSON.stringify(parsed.result)}`);
            } else if (parsed.error) {
              ensureOutput().appendLine(`✖ error: ${JSON.stringify(parsed.error)}`);
            }
          } catch { /* ignore non-JSON lines */ }
        }
      });
    });
    proc.stderr.on('data', (data: Buffer) => {
      ensureOutput().appendLine('[stderr] ' + data.toString());
    });
    proc.on('exit', (code) => {
      ensureOutput().appendLine(`[ext] MCP process exited code=${code}`);
      proc = null;
      ready = false;
    });
  });

  const stepCmd = vscode.commands.registerCommand('dlinoss.mcp.step', async () => {
    const stepsStr = await vscode.window.showInputBox({ prompt: 'Steps to advance', value: '128' });
    if (!stepsStr) return;
    const steps = parseInt(stepsStr, 10) || 1;
    sendRpc('dlinoss.step', { steps });
  });

  const fftCmd = vscode.commands.registerCommand('dlinoss.mcp.fft', async () => {
    const sizeStr = await vscode.window.showInputBox({ prompt: 'FFT size', value: '256' });
    if (!sizeStr) return;
    const size = parseInt(sizeStr, 10) || 256;
    sendRpc('dlinoss.getFft', { size });
  });

  context.subscriptions.push(startCmd, stepCmd, fftCmd);
}

export function deactivate() {
  if (proc) {
    proc.kill();
    proc = null;
  }
}
