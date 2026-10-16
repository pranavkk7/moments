// Moments page: choose a video, ask, and jump to the moments the answer cites.
const $ = (sel, root = document) => root.querySelector(sel)

const player = $('[data-player]')
const empty = $('[data-empty]')
const track = $('[data-timeline] .timeline-track')
const head = $('[data-head]')
const select = $('[data-videos]')
const result = $('[data-result]')
const now = $('[data-now]')

let videos = []
let current = null // the video loaded in the player
let marks = [] // citations for the loaded video
let answers = false // whether the server writes answers (it needs an LLM API key) or only finds moments
const askLabel = () => (answers ? 'Ask' : 'Find the moment')

function formatTime(seconds) {
  const s = Math.floor(seconds)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const pad = (n) => String(n).padStart(2, '0')
  return h ? `${h}:${pad(m)}:${pad(s % 60)}` : `${m}:${pad(s % 60)}`
}

async function api(path, options) {
  const response = await fetch(path, options)
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(body.detail || 'Something went wrong. Please try again.')
  return body
}

async function loadVideos(selectId) {
  videos = await api('/api/videos')
  select.replaceChildren(new Option('All videos', ''))
  for (const v of videos) select.add(new Option(`${v.title} (${v.language}, ${formatTime(v.duration)})`, v.id))
  if (selectId) {
    select.value = selectId
    play(selectId, 0)
  }
}

// Loads a video (if it isn't already) and starts it at `start` seconds.
function play(videoId, start) {
  if (!current || current.id !== videoId) {
    current = videos.find((v) => v.id === videoId)
    if (!current) return
    player.src = `/media/${encodeURIComponent(videoId)}`
    player.hidden = false
    empty.hidden = true
    showCredit()
  }
  const seek = () => {
    player.currentTime = start
    player.play().catch(() => {})
  }
  if (player.readyState >= 1) seek()
  else player.addEventListener('loadedmetadata', seek, { once: true })
  drawMarks()
}

// Sample videos carry their author and licence, shown under the player as their licences require.
function showCredit() {
  const credit = $('[data-credit]')
  credit.hidden = !current?.credit
  if (!current?.credit) return
  const link = Object.assign(document.createElement('a'), { href: current.credit.page, textContent: current.title, target: '_blank', rel: 'noopener' })
  credit.replaceChildren(link, ` by ${current.credit.author}, ${current.credit.license}, via Wikimedia Commons`)
}

// Yellow spans on the timeline, one per cited moment of the loaded video.
function drawMarks() {
  track.querySelectorAll('.timeline-mark').forEach((m) => m.remove())
  if (!current || !current.duration) return
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

function jump(citation) {
  play(citation.video_id, citation.start)
  $('[data-now-time]').textContent = `${citation.title} · ${formatTime(citation.start)}–${formatTime(citation.end)}`
  $('[data-now-text]').textContent = citation.text
  now.hidden = false
}

player.addEventListener('timeupdate', () => {
  if (current?.duration) head.style.left = `${(player.currentTime / current.duration) * 100}%`
})

select.addEventListener('change', () => {
  if (select.value) play(select.value, 0)
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
    text.textContent = marks.length ? 'Top moments' : 'Nothing in the video seems to match that question.'
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
      $('.cite-time', button).textContent = `${c.number}  ${formatTime(c.start)}`
      $('.cite-title', button).textContent = c.title
      $('.cite-text', button).textContent = c.text
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
    const { job_id } = await api('/api/videos', { method: 'POST', body: new FormData(form) })
    for (;;) {
      const job = await api(`/api/jobs/${job_id}`)
      meter.style.width = `${Math.round(job.progress * 100)}%`
      if (job.status === 'queued') status.textContent = 'Waiting for the previous video to finish…'
      if (job.status === 'transcribing') status.textContent = `Transcribing… ${Math.round(job.progress * 100)}%`
      if (job.status === 'failed') throw new Error(job.error)
      if (job.status === 'done') {
        status.textContent = 'Done. You can ask it questions now.'
        await loadVideos(job.video_id)
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

loadVideos().catch((error) => {
  result.replaceChildren(Object.assign(document.createElement('p'), { className: 'answer-error', textContent: error.message }))
})
