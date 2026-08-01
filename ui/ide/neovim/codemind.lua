-- CodeMind Neovim Plugin
-- ======================
-- Installation:
--   Copy to ~/.config/nvim/lua/codemind.lua
--   Add to init.lua:
--     require('codemind').setup({ api_key = 'your-key', server_url = 'http://localhost:8000' })
--
-- Keymaps (visual mode selection required for most):
--   <leader>cc  Complete at cursor
--   <leader>ce  Explain selected code
--   <leader>cf  Fix selected code
--   <leader>cd  Document selected code
--   <leader>cr  Refactor selected code
--   <leader>ct  Generate tests

local M = {}
local cfg = { api_key = '', server_url = 'http://localhost:8000', temperature = 0.3, max_tokens = 256 }

function M.setup(opts)
  cfg = vim.tbl_extend('force', cfg, opts or {})
  M._keymaps()
  vim.notify('CodeMind AI ready. Server: ' .. cfg.server_url, vim.log.levels.INFO)
end

local function post(endpoint, data, cb)
  local body = vim.fn.json_encode(data):gsub("'", "'\\''")
  local cmd = ("curl -s -X POST '%s%s' -H 'Content-Type: application/json' -H 'Authorization: Bearer %s' -d '%s'"):format(cfg.server_url, endpoint, cfg.api_key, body)
  local h = io.popen(cmd)
  local res = h:read('*a')
  h:close()
  local ok, dec = pcall(vim.fn.json_decode, res)
  if ok then cb(nil, dec) else cb('Parse error: ' .. res, nil) end
end

local function selection()
  local s = vim.fn.getpos("'<")
  local e = vim.fn.getpos("'>")
  local lines = vim.fn.getline(s[2], e[2])
  if #lines == 0 then return '' end
  lines[#lines] = lines[#lines]:sub(1, e[3])
  lines[1] = lines[1]:sub(s[3])
  return table.concat(lines, '\n')
end

local function lang() return vim.bo.filetype or 'python' end

local function open_buf(title, content)
  local buf = vim.api.nvim_create_buf(false, true)
  vim.api.nvim_buf_set_name(buf, title)
  vim.api.nvim_buf_set_lines(buf, 0, -1, false, vim.split(content, '\n'))
  vim.api.nvim_buf_set_option(buf, 'modifiable', false)
  vim.api.nvim_buf_set_option(buf, 'filetype', lang())
  vim.cmd('vsplit')
  vim.api.nvim_win_set_buf(vim.api.nvim_get_current_win(), buf)
end

function M.complete()
  local row = vim.api.nvim_win_get_cursor(0)[1]
  local lines = vim.api.nvim_buf_get_lines(0, 0, row, false)
  post('/v1/ide/complete', { prefix = table.concat(lines, '\n'), language = lang(), max_tokens = cfg.max_tokens, temperature = cfg.temperature }, function(err, data)
    if err then vim.notify('CodeMind: ' .. err, vim.log.levels.ERROR); return end
    local c = (data or {}).completion or ''
    if c ~= '' then vim.api.nvim_put(vim.split(c, '\n'), 'c', true, true) end
  end)
end

function M.explain()
  local code = selection()
  if code == '' then vim.notify('Select code first', vim.log.levels.WARN); return end
  post('/v1/ide/explain', { code = code, language = lang() }, function(err, data)
    if err then vim.notify('CodeMind: ' .. err, vim.log.levels.ERROR); return end
    open_buf('CodeMind: Explanation', (data or {}).explanation or '')
  end)
end

function M.fix()
  local code = selection()
  if code == '' then vim.notify('Select code first', vim.log.levels.WARN); return end
  local err_msg = vim.fn.input('Error: ')
  post('/v1/ide/fix', { code = code, error = err_msg, language = lang() }, function(err, data)
    if err then vim.notify('CodeMind: ' .. err, vim.log.levels.ERROR); return end
    open_buf('CodeMind: Fixed', (data or {}).fixed_code or '')
  end)
end

function M.document()
  local code = selection()
  if code == '' then vim.notify('Select code first', vim.log.levels.WARN); return end
  post('/v1/ide/document', { code = code, language = lang() }, function(err, data)
    if err then vim.notify('CodeMind: ' .. err, vim.log.levels.ERROR); return end
    open_buf('CodeMind: Documented', (data or {}).documented_code or '')
  end)
end

function M.refactor()
  local code = selection()
  if code == '' then vim.notify('Select code first', vim.log.levels.WARN); return end
  local instr = vim.fn.input('How to refactor: ')
  post('/v1/ide/refactor', { code = code, instruction = instr, language = lang() }, function(err, data)
    if err then vim.notify('CodeMind: ' .. err, vim.log.levels.ERROR); return end
    open_buf('CodeMind: Refactored', (data or {}).refactored_code or '')
  end)
end

function M.test()
  local code = selection()
  if code == '' then vim.notify('Select code first', vim.log.levels.WARN); return end
  post('/v1/ide/test', { code = code, language = lang(), framework = 'pytest' }, function(err, data)
    if err then vim.notify('CodeMind: ' .. err, vim.log.levels.ERROR); return end
    open_buf('CodeMind: Tests', (data or {}).tests or '')
  end)
end

function M._keymaps()
  local o = { noremap = true, silent = true }
  vim.keymap.set('n', '<leader>cc', M.complete,  vim.tbl_extend('force', o, { desc = 'CodeMind: Complete' }))
  vim.keymap.set('v', '<leader>ce', M.explain,   vim.tbl_extend('force', o, { desc = 'CodeMind: Explain' }))
  vim.keymap.set('v', '<leader>cf', M.fix,       vim.tbl_extend('force', o, { desc = 'CodeMind: Fix' }))
  vim.keymap.set('v', '<leader>cd', M.document,  vim.tbl_extend('force', o, { desc = 'CodeMind: Document' }))
  vim.keymap.set('v', '<leader>cr', M.refactor,  vim.tbl_extend('force', o, { desc = 'CodeMind: Refactor' }))
  vim.keymap.set('v', '<leader>ct', M.test,      vim.tbl_extend('force', o, { desc = 'CodeMind: Test' }))
end

return M
