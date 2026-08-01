
const vscode = require('vscode');

function getConfig() {
  const c = vscode.workspace.getConfiguration('codemind');
  return {
    serverUrl:    c.get('serverUrl')    || 'http://localhost:8000',
    apiKey:       c.get('apiKey')       || '',
    temperature:  c.get('temperature')  || 0.3,
    maxTokens:    c.get('maxTokens')    || 256,
    autoComplete: c.get('autoComplete') ?? true,
  };
}

async function apiPost(endpoint, data) {
  const { serverUrl, apiKey } = getConfig();
  const res = await fetch(`${serverUrl}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
    body: JSON.stringify(data),
  });
  if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail || `HTTP ${res.status}`); }
  return res.json();
}

function getCtx() {
  const ed = vscode.window.activeTextEditor;
  if (!ed) return null;
  const sel   = ed.selection;
  const selTx = ed.document.getText(sel);
  const prefix = ed.document.getText(new vscode.Range(new vscode.Position(0,0), sel.start));
  const suffix = ed.document.getText(new vscode.Range(sel.end, new vscode.Position(ed.document.lineCount,0)));
  return { selectedText: selTx, prefix, suffix, language: ed.document.languageId, editor: ed };
}

async function withProgress(title, fn) {
  return vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title }, fn);
}

async function cmdComplete() {
  const ctx = getCtx(); if (!ctx) return;
  await withProgress('CodeMind: Completing...', async () => {
    try {
      const cfg = getConfig();
      const r = await apiPost('/v1/ide/complete', { prefix: ctx.prefix, suffix: ctx.suffix, language: ctx.language, max_tokens: cfg.maxTokens, temperature: cfg.temperature });
      if (r.completion) await ctx.editor.edit(eb => eb.insert(ctx.editor.selection.active, r.completion));
    } catch(e) { vscode.window.showErrorMessage(`CodeMind: ${e.message}`); }
  });
}

async function cmdExplain() {
  const ctx = getCtx();
  if (!ctx || !ctx.selectedText) { vscode.window.showWarningMessage('Select code to explain.'); return; }
  await withProgress('CodeMind: Explaining...', async () => {
    try {
      const r = await apiPost('/v1/ide/explain', { code: ctx.selectedText, language: ctx.language });
      const panel = vscode.window.createWebviewPanel('cmExplain', 'CodeMind: Explanation', vscode.ViewColumn.Beside, {});
      panel.webview.html = `<html><body style="font-family:sans-serif;padding:20px;background:#1e1e1e;color:#d4d4d4"><h2 style="color:#569cd6">Explanation</h2><pre style="white-space:pre-wrap;background:#252526;padding:16px;border-radius:8px">${r.explanation || ''}</pre></body></html>`;
    } catch(e) { vscode.window.showErrorMessage(`CodeMind: ${e.message}`); }
  });
}

async function cmdFix() {
  const ctx = getCtx();
  if (!ctx || !ctx.selectedText) { vscode.window.showWarningMessage('Select code to fix.'); return; }
  const error = await vscode.window.showInputBox({ prompt: 'Describe the error:' });
  if (!error) return;
  await withProgress('CodeMind: Fixing...', async () => {
    try {
      const r = await apiPost('/v1/ide/fix', { code: ctx.selectedText, error, language: ctx.language });
      if (r.fixed_code) await ctx.editor.edit(eb => eb.replace(ctx.editor.selection, r.fixed_code));
      vscode.window.showInformationMessage('CodeMind: Fixed!');
    } catch(e) { vscode.window.showErrorMessage(`CodeMind: ${e.message}`); }
  });
}

async function cmdDocument() {
  const ctx = getCtx();
  if (!ctx || !ctx.selectedText) { vscode.window.showWarningMessage('Select code to document.'); return; }
  await withProgress('CodeMind: Documenting...', async () => {
    try {
      const r = await apiPost('/v1/ide/document', { code: ctx.selectedText, language: ctx.language });
      if (r.documented_code) await ctx.editor.edit(eb => eb.replace(ctx.editor.selection, r.documented_code));
      vscode.window.showInformationMessage('CodeMind: Documented!');
    } catch(e) { vscode.window.showErrorMessage(`CodeMind: ${e.message}`); }
  });
}

async function cmdRefactor() {
  const ctx = getCtx();
  if (!ctx || !ctx.selectedText) { vscode.window.showWarningMessage('Select code to refactor.'); return; }
  const instruction = await vscode.window.showInputBox({ prompt: 'How to refactor?' });
  if (!instruction) return;
  await withProgress('CodeMind: Refactoring...', async () => {
    try {
      const r = await apiPost('/v1/ide/refactor', { code: ctx.selectedText, instruction, language: ctx.language });
      if (r.refactored_code) await ctx.editor.edit(eb => eb.replace(ctx.editor.selection, r.refactored_code));
      vscode.window.showInformationMessage('CodeMind: Refactored!');
    } catch(e) { vscode.window.showErrorMessage(`CodeMind: ${e.message}`); }
  });
}

async function cmdTest() {
  const ctx = getCtx();
  if (!ctx || !ctx.selectedText) { vscode.window.showWarningMessage('Select code to test.'); return; }
  const frameworks = { python:['pytest','unittest'], javascript:['jest','mocha'], typescript:['jest','vitest'], java:['junit'], go:['testing'], rust:['cargo test'] };
  const opts = frameworks[ctx.language] || ['pytest'];
  const framework = await vscode.window.showQuickPick(opts, { placeHolder: 'Test framework' });
  if (!framework) return;
  await withProgress('CodeMind: Generating tests...', async () => {
    try {
      const r = await apiPost('/v1/ide/test', { code: ctx.selectedText, framework, language: ctx.language });
      const doc = await vscode.workspace.openTextDocument({ content: r.tests || '', language: ctx.language });
      await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
    } catch(e) { vscode.window.showErrorMessage(`CodeMind: ${e.message}`); }
  });
}

async function cmdChat() {
  const { serverUrl, apiKey } = getConfig();
  const panel = vscode.window.createWebviewPanel('cmChat', '👑 CodeMind Chat', vscode.ViewColumn.Beside, { enableScripts: true, retainContextWhenHidden: true });
  panel.webview.html = `<!DOCTYPE html>
<html><head><style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#1e1e1e;color:#d4d4d4;font-family:'Segoe UI',sans-serif;height:100vh;display:flex;flex-direction:column}
#hdr{background:#252526;padding:12px 16px;border-bottom:1px solid #3e3e42}
h1{color:#569cd6;font-size:1.1em}
#msgs{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px}
.msg{padding:10px 14px;border-radius:8px;max-width:85%;line-height:1.5}
.user{background:#264f78;align-self:flex-end}
.assistant{background:#252526;border:1px solid #3e3e42;align-self:flex-start}
pre{background:#1e1e1e;padding:10px;border-radius:4px;overflow-x:auto;margin-top:8px;font-size:.85em}
#row{display:flex;gap:8px;padding:12px;border-top:1px solid #3e3e42;background:#252526}
#inp{flex:1;background:#3e3e42;border:1px solid #555;color:#d4d4d4;padding:8px 12px;border-radius:6px;font-size:.9em;resize:none;height:60px}
button{background:#007acc;color:#fff;border:none;padding:8px 16px;border-radius:6px;cursor:pointer;font-weight:bold}
button:hover{background:#0098ff}
</style></head>
<body>
<div id="hdr"><h1>&#128081; CodeMind AI Chat</h1></div>
<div id="msgs"><div class="msg assistant">Hello! I am CodeMind — your standalone AI. How can I help?</div></div>
<div id="row">
  <textarea id="inp" placeholder="Ask anything... (Enter to send, Shift+Enter for newline)"></textarea>
  <button onclick="send()">Send</button>
</div>
<script>
const SERVER='${serverUrl}', KEY='${apiKey}';
const msgs=document.getElementById('msgs'), inp=document.getElementById('inp');
const history=[];
inp.addEventListener('keydown',e=>{ if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();} });
async function send(){
  const txt=inp.value.trim(); if(!txt) return; inp.value='';
  add('user',txt); history.push({role:'user',content:txt});
  const el=add('assistant','Thinking...'); el.style.color='#888';
  try{
    const r=await fetch(SERVER+'/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+KEY},body:JSON.stringify({model:'codemind-v2',messages:history,max_tokens:2048})});
    const d=await r.json();
    const c=d.choices?.[0]?.message?.content||'No response';
    el.style.color=''; el.innerHTML=fmt(c); history.push({role:'assistant',content:c});
  }catch(e){ el.textContent='Error: '+e.message; }
  msgs.scrollTop=msgs.scrollHeight;
}
function add(role,content){ const el=document.createElement('div'); el.className='msg '+role; el.innerHTML=fmt(content); msgs.appendChild(el); msgs.scrollTop=msgs.scrollHeight; return el; }
function fmt(t){ return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\`\`\`([\s\S]*?)\`\`\`/g,'<pre><code>$1</code></pre>').replace(/\`([^\`]+)\`/g,'<code style="background:#3e3e42;padding:1px 4px;border-radius:3px">$1</code>').replace(/\n/g,'<br>'); }
</script></body></html>`;
}

function activate(context) {
  console.log('CodeMind AI activated');
  const cmds = [
    vscode.commands.registerCommand('codemind.complete', cmdComplete),
    vscode.commands.registerCommand('codemind.explain',  cmdExplain),
    vscode.commands.registerCommand('codemind.fix',      cmdFix),
    vscode.commands.registerCommand('codemind.document', cmdDocument),
    vscode.commands.registerCommand('codemind.refactor', cmdRefactor),
    vscode.commands.registerCommand('codemind.test',     cmdTest),
    vscode.commands.registerCommand('codemind.chat',     cmdChat),
  ];
  cmds.forEach(c => context.subscriptions.push(c));

  const bar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  bar.text = '$(robot) CodeMind'; bar.command = 'codemind.chat';
  bar.tooltip = 'Open CodeMind AI'; bar.show();
  context.subscriptions.push(bar);

  const cfg = getConfig();
  if (cfg.autoComplete) {
    const provider = vscode.languages.registerInlineCompletionItemProvider({ pattern: '**' }, {
      async provideInlineCompletionItems(doc, pos) {
        const prefix = doc.getText(new vscode.Range(new vscode.Position(0,0), pos));
        if (prefix.length < 10) return [];
        try {
          const { apiKey, serverUrl, temperature, maxTokens } = getConfig();
          const r = await fetch(serverUrl+'/v1/ide/complete', {
            method:'POST', signal: AbortSignal.timeout(5000),
            headers:{'Content-Type':'application/json','Authorization':'Bearer '+apiKey},
            body: JSON.stringify({ prefix, language: doc.languageId, max_tokens: maxTokens, temperature }),
          });
          const d = await r.json();
          if (!d.completion) return [];
          return [new vscode.InlineCompletionItem(d.completion, new vscode.Range(pos,pos))];
        } catch { return []; }
      }
    });
    context.subscriptions.push(provider);
  }
}

function deactivate() {}
module.exports = { activate, deactivate };
