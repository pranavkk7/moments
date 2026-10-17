// Moments page: choose a video or a document, ask, and open the moments the search found: a video at
// the timestamp, a PDF at the page.
const $ = (sel, root = document) => root.querySelector(sel)

const player = $('[data-player]')
const docFrame = $('[data-doc]')
const screen = $('.screen')
const empty = $('[data-empty]')
const timeline = $('[data-timeline]')
const track = $('[data-timeline] .timeline-track')
const head = $('[data-head]')
const select = $('[data-videos]')
const result = $('[data-result]')
const now = $('[data-now]')

let items = []
let current = null // the video or document on show
let marks = [] // the moments found for the last question
let answers = false // whether the server writes answers (it needs an LLM API key) or only finds moments
const askLabel = () => (answers ? 'Ask' : 'Find the moment')

function formatTime(seconds) {
  const s = Math.floor(seconds)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const pad = (n) => String(n).padStart(2, '0')
  return h ? `${h}:${pad(m)}:${pad(s % 60)}` : `${m}:${pad(s % 60)}`
}

const isDocument = (item) => item?.kind === 'document'

// Where a moment is: a timestamp in a video, a page (or a passage number) in a document.
function place(c) {
  if (c.kind !== 'document') return formatTime(c.start)
  return c.page ? `p. ${c.page}` : `part ${c.part}`
}

function label(item) {
  if (!isDocument(item)) return `${item.title} (${item.language}, ${formatTime(item.duration)})`
  const pages = item.pages ? `, ${item.pages} pages` : ''
  return `${item.title} (${item.format.toUpperCase()}, ${item.language}${pages})`
}

async function api(path, options) {
  const response = await fetch(path, options)
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(body.detail || 'Something went wrong. Please try again.')
  return body
}

async function loadItems(selectId) {
  items = await api('/api/items')
  select.replaceChildren(new Option('Everything', ''))
  for (const item of items) select.add(new Option(label(item), item.id))
  if (selectId) {
    select.value = selectId
    showItem(selectId, {})
  }
}

// Shows a video at `start` seconds, or a document at `page`.
function showItem(itemId, { start = 0, page = null } = {}) {
  const item = items.find((i) => i.id === itemId)
  if (!item) return
  const changed = current?.id !== itemId
  current = item
  empty.hidden = true
  if (changed) showCredit()
  if (isDocument(item)) showDocument(item, page)
  else play(item, start, changed)
  drawMarks()
}

function play(item, start, changed) {
  docFrame.hidden = true
  $('[data-doc-link]').hidden = true
  screen.classList.remove('is-doc')
  timeline.hidden = false
  player.hidden = false
  if (changed) player.src = `/media/${encodeURIComponent(item.id)}`
  const seek = () => {
    player.currentTime = start
    player.play().catch(() => {})
  }
  if (player.readyState >= 1 && !changed) seek()
  else player.addEventListener('loadedmetadata', seek, { once: true })
}

// PDFs open in the browser's own viewer at the page; Word and text files have no pages to show, so the
// passage below the viewer is the place.
function showDocument(item, page) {
  player.pause()
  player.hidden = true
  timeline.hidden = true
  screen.classList.add('is-doc')
  // A link to the same page in a new tab, for browsers set to download PDFs rather than show them.
  const link = $('[data-doc-link] a')
  $('[data-doc-link]').hidden = false
  if (item.format === 'pdf') {
    const target = page || 1
    // The query string makes the viewer reload when only the page changes.
    docFrame.src = `/media/${encodeURIComponent(item.id)}?page=${target}#page=${target}`
    docFrame.hidden = false
    link.href = docFrame.src
    link.textContent = `Open page ${target} in a new tab`
  } else {
    link.href = `/media/${encodeURIComponent(item.id)}`
    link.textContent = 'Open the file in a new tab'
    docFrame.hidden = true
    empty.hidden = false
    $('.empty-title', empty).textContent = item.title
    $('.empty-text', empty).textContent = 'This file has no pages to show. The matching passage is below.'
  }
}

// Sample videos carry their author and licence, shown under the player as their licences require.
function showCredit() {
  const credit = $('[data-credit]')
  credit.hidden = !current?.credit
  if (!current?.credit) return
  const link = Object.assign(document.createElement('a'), { href: current.credit.page, textContent: current.title, target: '_blank', rel: 'noopener' })
  credit.replaceChildren(link, ` by ${current.credit.author}, ${current.credit.license}, via Wikimedia Commons`)
}

// Yellow spans on the timeline, one per moment found in the video on show.
function drawMarks() {
  track.querySelectorAll('.timeline-mark').forEach((m) => m.remove())
  if (!current || isDocument(current) || !current.duration) return
  marks.forEach((c) => {
    if (c.video_id !== current.id) return
    const mark = document.createElement('button')
    mark.type = 'button'
    mark.className = 'timeline-mark'
    mark.textContent = c.number
    mark.style.left = `${(c.start / current.duration) * 100}%`
    mark.style.width = `${((c.end - c.start) / current.duration) * 100}%`
    mark.setAttribute('aria-label', `Moment ${c.number} at ${formatTime(c.start)}`)
    mark.addEventListener('click', () => jump(c))
    track.append(mark)
  })
}

// Writes `text` into `el` with the best-matching sentence marked, built with text nodes so nothing in
// a transcript or document is ever treated as HTML.
function writeMarked(el, text, best) {
  el.replaceChildren()
  const at = best ? text.indexOf(best) : -1
  if (at < 0) {
    el.textContent = text
    return
  }
  el.append(text.slice(0, at), Object.assign(document.createElement('mark'), { textContent: best }), text.slice(at + best.length))
}

function jump(citation) {
  showItem(citation.video_id, { start: citation.start, page: citation.page })
  const where = citation.kind === 'document'
    ? (citation.page ? `page ${citation.page}` : `passage ${citation.part}`)
    : `${formatTime(citation.start)}–${formatTime(citation.end)}`
  $('[data-now-time]').textContent = `${citation.title} · ${where}`
  writeMarked($('[data-now-text]'), citation.english || citation.text, citation.best)
  const original = $('[data-now-original]')
  original.textContent = citation.english ? citation.text : ''
  original.hidden = !citation.english
  now.hidden = false
}

player.addEventListener('timeupdate', () => {
  if (current?.duration) head.style.left = `${(player.currentTime / current.duration) * 100}%`
})

select.addEventListener('change', () => {
  if (select.value) showItem(select.value, {})
})

$('[data-ask]').addEventListener('submit', async (event) => {
  event.preventDefault()
  const question = $('[data-question]').value.trim()
  if (!question) return
  const button = $('[data-ask-button]')
  button.disabled = true
  button.textContent = 'Finding the moment…'
  try {
    const body = await api('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, video_id: select.value || null }),
    })
    marks = body.citations.map((c, i) => ({ ...c, number: i + 1 }))
    showAnswer(body)
    if (marks.length) jump(marks[0])
    else drawMarks()
  } catch (error) {
    result.replaceChildren(Object.assign(document.createElement('p'), { className: 'answer-error', textContent: error.message }))
  } finally {
    button.disabled = false
    button.textContent = askLabel()
  }
})

function showAnswer(body) {
  const box = document.createElement('div')
  box.className = 'answer'
  const text = document.createElement('p')
  if (body.written || body.answer) {
    text.className = body.found ? 'answer-text' : 'answer-text answer-missing'
    text.textContent = body.answer
  } else {
    // Search only: the moments are the answer.
    text.className = 'answer-heading'
    text.textContent = marks.length ? 'Top moments' : 'Nothing in the library seems to match that question.'
  }
  box.append(text)
  if (marks.length) {
    const list = document.createElement('ol')
    list.className = 'cites'
    for (const c of marks) {
      const item = document.createElement('li')
      const button = document.createElement('button')
      button.type = 'button'
      button.className = 'cite'
      button.innerHTML = '<span class="cite-time"></span><span class="cite-title"></span><span class="cite-text"></span>'
      $('.cite-time', button).textContent = `${c.number}  ${place(c)}`
      $('.cite-title', button).textContent = c.title
      // English first for a video in another language; the original words stay underneath.
      $('.cite-text', button).textContent = c.best || c.english || c.text
      if (c.english) {
        const original = Object.assign(document.createElement('span'), { className: 'cite-original', textContent: c.text })
        button.append(original)
      }
      button.addEventListener('click', () => jump(c))
      item.append(button)
      list.append(item)
    }
    box.append(list)
  }
  result.replaceChildren(box)
}

// ---------- upload ----------

const dialog = $('[data-upload]')
const form = $('[data-upload-form]')
$('[data-open-upload]').addEventListener('click', () => dialog.showModal())
$('[data-close-upload]').addEventListener('click', () => dialog.close())

form.addEventListener('submit', async (event) => {
  event.preventDefault()
  const button = $('[data-upload-button]')
  const status = $('[data-upload-status]')
  const meter = $('[data-meter]')
  button.disabled = true
  $('[data-upload-progress]').hidden = false
  status.textContent = 'Uploading…'
  meter.style.width = '0'
  try {
    const { job_id } = await api('/api/items', { method: 'POST', body: new FormData(form) })
    for (;;) {
      const job = await api(`/api/jobs/${job_id}`)
      meter.style.width = `${Math.round(job.progress * 100)}%`
      if (job.status === 'queued') status.textContent = 'Waiting for the previous file to finish…'
      if (job.status === 'reading') status.textContent = 'Reading the document…'
      if (job.status === 'transcribing') status.textContent = `Transcribing… ${Math.round(job.progress * 100)}%`
      if (job.status === 'translating') status.textContent = `Translating into English… ${Math.round(job.progress * 100)}%`
      if (job.status === 'failed') throw new Error(job.error)
      if (job.status === 'done') {
        status.textContent = 'Done. You can ask it questions now.'
        await loadItems(job.video_id)
        setTimeout(() => dialog.close(), 900)
        form.reset()
        break
      }
      await new Promise((r) => setTimeout(r, 1500))
    }
  } catch (error) {
    status.textContent = error.message
  } finally {
    button.disabled = false
  }
})

api('/api/config').then((config) => {
  $('[data-open-upload]').hidden = !config.uploads
  answers = config.answers
  $('[data-ask-button]').textContent = askLabel()
}).catch(() => {})

loadItems().catch((error) => {
  result.replaceChildren(Object.assign(document.createElement('p'), { className: 'answer-error', textContent: error.message }))
})
