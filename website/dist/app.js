(() => {
  'use strict';
  const D = window.ELEC_DATA;
  if (!D) return;
  const $ = id => document.getElementById(id);
  const taskNames = {
    T01: 'Conditional connection', T02: 'Minimum closure', T03: 'Wire-only grouping',
    T04: 'Opening a contact', T05: 'Unknown contact state', T06: 'Hypothetical bypass',
    T07: 'Fault candidates', T08: 'Measurement planning', T09: 'Design check',
    T10: 'Repair check', T11: 'Contact order', T12: 'Series or parallel',
    T13: 'Diagnosis sufficiency', T14: 'Information sufficiency', T15: 'Branch interruption'
  };
  const displayName = {base:'Base', A:'A', E:'E', internvl:'InternVL', llava:'LLaVA'};
  const colors = {base:'#8aa69b', A:'#86b5b0', E:'#e1aa7e', internvl:'#7a998e', llava:'#9ab29d'};
  const pct = v => (Number(v) * 100).toFixed(2);
  const el = (tag, className, text) => {
    const n = document.createElement(tag);
    if (className) n.className = className;
    if (text !== undefined) n.textContent = text;
    return n;
  };

  // Real data browser. No model inference is performed in the page.
  let example = 0;
  const exampleButtons = D.examples.map((r, i) => {
    const b = el('button', '', taskNames[r.task.slice(0,3)] || r.task);
    b.prepend(el('small', '', `${String(i + 1).padStart(2,'0')} / ${r.task.slice(0,3)}`));
    b.addEventListener('click', () => showExample(i));
    $('example-list').append(b);
    return b;
  });
  function showExample(i) {
    example = (i + D.examples.length) % D.examples.length;
    const r = D.examples[example];
    exampleButtons.forEach((b, j) => {
      b.classList.toggle('active', j === example);
      b.setAttribute('aria-pressed', String(j === example));
    });
    $('example-image').src = example === 0 ? 'assets/example_01_detail.png' : r.image;
    $('example-image').alt = `Original schematic for ${taskNames[r.task.slice(0,3)]}`;
    $('example-type').textContent = `${r.task.slice(0,3)} / ${taskNames[r.task.slice(0,3)]}`;
    $('example-id').textContent = r.id;
    $('example-question').textContent = r.question;
    $('example-answer').textContent = r.answer;
    $('example-evidence').textContent = r.explanation;
    $('example-evidence').hidden = true;
    $('evidence-toggle').setAttribute('aria-expanded','false');
    $('evidence-toggle').firstElementChild.textContent = 'Reveal compact evidence';
    $('evidence-toggle').lastElementChild.textContent = '＋';
    $('example-count').textContent = `${example + 1} / ${D.examples.length}`;
  }
  $('example-prev').addEventListener('click', () => showExample(example - 1));
  $('example-next').addEventListener('click', () => showExample(example + 1));
  $('evidence-toggle').addEventListener('click', () => {
    const open = $('example-evidence').hidden;
    $('example-evidence').hidden = !open;
    $('evidence-toggle').setAttribute('aria-expanded', String(open));
    $('evidence-toggle').firstElementChild.textContent = open ? 'Hide compact evidence' : 'Reveal compact evidence';
    $('evidence-toggle').lastElementChild.textContent = open ? '−' : '＋';
  });
  showExample(0);

  function zoom(image, description) {
    $('image-viewport').classList.remove('native');
    $('native-size').textContent = 'Show original pixels';
    $('dialog-image').src = image;
    $('dialog-image').alt = description || 'Enlarged original electrical schematic';
    $('image-dialog').showModal();
  }
  document.querySelectorAll('.zoom-trigger').forEach(b => b.addEventListener('click', () => zoom(b.dataset.image)));
  $('example-zoom').addEventListener('click', () => zoom(D.examples[example].image));
  $('example-picture').addEventListener('click', () => zoom(D.examples[example].image));
  $('close-dialog').addEventListener('click', () => $('image-dialog').close());
  $('native-size').addEventListener('click', () => {
    const native = $('image-viewport').classList.toggle('native');
    $('native-size').textContent = native ? 'Fit to window' : 'Show original pixels';
  });
  $('image-dialog').addEventListener('click', e => {if(e.target === $('image-dialog')) $('image-dialog').close();});

  // Aligned training target views use the exact reference example.
  function showTarget(view) {
    const container = $('target-content');
    container.replaceChildren();
    ['a','e'].forEach(k => {
      $(`target-${k}`).classList.toggle('active',k === view);
      $(`target-${k}`).setAttribute('aria-selected',String(k === view));
      $(`target-${k}`).tabIndex = k === view ? 0 : -1;
    });
    if (view === 'e') {
      container.append(el('span','target-label','COMPACT EVIDENCE'));
      container.append(el('p','',D.examples[0].explanation));
    }
    container.append(el('span','target-label','ANSWER'));
    container.append(el('p', view === 'e' ? 'target-answer' : '', D.examples[0].answer));
  }
  ['a','e'].forEach(k => {
    $(`target-${k}`).addEventListener('click', () => showTarget(k));
    $(`target-${k}`).addEventListener('keydown', e => {
      if (['ArrowLeft','ArrowRight'].includes(e.key)) {
        e.preventDefault();const next = k === 'a' ? 'e' : 'a';showTarget(next);$(`target-${next}`).focus();
      }
    });
  });
  showTarget('e');

  // The same source numbers drive all interactive result states.
  let track = 'explain', metric = 'joint';
  function score(model) {
    if ($('supplemented').checked) {
      if (metric === 'joint') return Number(D.metrics.joint.find(r => r.model === model).joint);
      return D.metrics.summary.find(r => r.track === track && r.dimension === metric && r.model === model).accuracy;
    }
    if (metric === 'joint' || metric === 'evidence') {
      return Number(D.original_joint.find(r => r.model === model)[metric]);
    }
    return Number(D.original_overall.find(r => r.track === track && r.model === model).accuracy);
  }
  function renderChart() {
    const supplemented = $('supplemented').checked;
    document.querySelectorAll('[data-track]').forEach(b => {
      b.classList.toggle('active', b.dataset.track === track);
      b.setAttribute('aria-pressed',String(b.dataset.track === track));
    });
    document.querySelectorAll('[data-metric]').forEach(b => {
      b.disabled = track === 'direct' && b.dataset.metric !== 'answer';
      b.classList.toggle('active',b.dataset.metric === metric);
      b.setAttribute('aria-pressed',String(b.dataset.metric === metric));
    });
    const models = track === 'direct' ? ['base','internvl','llava','A','E'] : ['base','A','E'];
    const bars = models.map(m => {
      const value = score(m), row = el('div','chart-row');
      row.setAttribute('role','listitem');row.setAttribute('aria-label',`${displayName[m]}: ${pct(value)} percent full credit`);
      const label = el('span','chart-model',displayName[m]);
      const bg = el('div','bar-track'), bar = el('div','bar-value');
      bar.style.setProperty('--value',`${value * 100}%`);bar.style.setProperty('--fill',colors[m]);
      bar.append(el('span','',pct(value)));bg.append(bar);row.append(label,bg);return row;
    });
    $('chart-bars').setAttribute('role','list');$('chart-bars').replaceChildren(...bars);
    const addResult = (number,unit) => {$('chart-result').replaceChildren(document.createTextNode(number),el('span','',unit));};
    if (track === 'direct') {
      addResult('+'+((score('A')-score('base'))*100).toFixed(2),'pp');
      $('chart-label').textContent = 'DOMAIN ADAPTATION · A MINUS BASE';
      $('chart-title').textContent = 'Domain data improves direct answers.';
      $('chart-description').textContent = 'Answer-only adaptation is the strongest direct-answer model in this comparison. E remains above Base, but below A.';
    } else if (metric === 'joint') {
      addResult('+'+((score('E')-score('A'))*100).toFixed(2),'pp');
      $('chart-label').textContent = 'SAME-RESPONSE JOINT FULL CREDIT';
      $('chart-title').textContent = 'More answers with valid support.';
      $('chart-description').textContent = 'With a common explanation prompt, E improves joint answer–evidence full credit over A. This is a gain in supported output, not universal answer accuracy.';
    } else if (metric === 'evidence') {
      addResult('+'+((score('E')-score('A'))*100).toFixed(2),'pp');
      $('chart-label').textContent = 'EXPLICIT EVIDENCE · E MINUS A';
      $('chart-title').textContent = 'Stronger observable justification.';
      $('chart-description').textContent = 'Evidence must be relevant, electrically valid and sufficient under the stated conditions. A correct answer alone does not earn evidence credit.';
    } else {
      addResult('+'+((score('A')-score('E'))*100).toFixed(2),'pp');
      $('chart-label').textContent = 'ANSWER FULL CREDIT · A MINUS E';
      $('chart-title').textContent = 'E does not win every comparison.';
      $('chart-description').textContent = 'A retains the higher point estimate for final answers. The source-bootstrap interval for the A–E difference spans zero; this does not establish equivalence.';
    }
    $('chart-footnote').textContent = supplemented ? 'Primary test: 2,064 questions. Includes provisional supplementary ratings; human adjudication pending.' : 'Original Qwen-only full-credit lower estimates. Unresolved items remain in the denominator; no supplementary scores.';
  }
  document.querySelectorAll('[data-track]').forEach(b => b.addEventListener('click', () => {track=b.dataset.track;if(track==='direct') metric='answer';renderChart();}));
  document.querySelectorAll('[data-metric]').forEach(b => b.addEventListener('click', () => {metric=b.dataset.metric;renderChart();}));
  $('supplemented').addEventListener('change',renderChart);
  renderChart();

  const pairCopy = {
    state_flip:['When a changed contact should change the decision.','E is more successful on this subset of relevant contact-state changes. Both answers must be correct.'],
    state_invariance:['When the answer should stay the same.','E often changes or misjudges a decision even when the changed contact should not affect it. Condition relevance remains a major weakness.'],
    visual_invariance:['A drawing change that preserves the decision.','E exceeds A on this small invariant subset. These are controlled drawing pairs, not independent new industrial designs.'],
    visual_flip:['A topology change that should reverse the answer.','Both adapters struggle, and E is weaker. Good verbal evidence does not guarantee accurate perception of altered wiring.']
  };
  function showPair(kind) {
    const rows = ['A','E'].map(m => D.pairs.find(r => r.track==='explain' && r.model===m && r.kind===kind));
    document.querySelectorAll('[data-pair]').forEach(b => {b.classList.toggle('active',b.dataset.pair===kind);b.setAttribute('aria-pressed',String(b.dataset.pair===kind));});
    $('pair-title').textContent = pairCopy[kind][0];$('pair-description').textContent = pairCopy[kind][1];
    $('pair-count').textContent = `${rows[0].n} CONTROLLED PAIRS · EXPLANATION REQUESTED`;
    $('pair-bars').replaceChildren(...rows.map(r => {
      const row=el('div','pair-bar');row.append(el('span','',r.model));
      const bg=el('div'),bar=el('i');bar.style.setProperty('--value',`${Number(r.lower)*100}%`);bar.style.setProperty('--fill',r.model==='A'?'#245b78':'#cf6a39');bg.append(bar);
      row.append(bg,el('b','',pct(r.lower)+'%'));row.setAttribute('aria-label',`${r.model}: ${r.both_correct} of ${r.n} pairs correct`);return row;
    }));
  }
  document.querySelectorAll('[data-pair]').forEach(b=>b.addEventListener('click',()=>showPair(b.dataset.pair)));
  showPair('state_flip');

  function showCase(mode) {
    $('case-a').textContent=D.case_outputs[`${mode}_A`];$('case-e').textContent=D.case_outputs[`${mode}_E`];
    document.querySelectorAll('[data-case]').forEach(b=>{b.classList.toggle('active',b.dataset.case===mode);b.setAttribute('aria-pressed',String(b.dataset.case===mode));});
  }
  document.querySelectorAll('[data-case]').forEach(b=>b.addEventListener('click',()=>showCase(b.dataset.case)));
  showCase('explain');

  $('copy-citation').addEventListener('click',async()=>{
    try {
      await navigator.clipboard.writeText($('bibtex').textContent);
      $('copy-status').textContent='Copied to clipboard.';
    } catch {
      const selection=window.getSelection(),range=document.createRange();range.selectNodeContents($('bibtex'));selection.removeAllRanges();selection.addRange(range);
      $('copy-status').textContent='Citation selected. Copy it with Ctrl/Cmd+C.';
    }
  });
  const menu=document.querySelector('.menu-toggle');
  menu.addEventListener('click',()=>{const open=menu.getAttribute('aria-expanded')!=='true';menu.setAttribute('aria-expanded',String(open));$('nav').classList.toggle('open',open);});
  $('nav').querySelectorAll('a').forEach(a=>a.addEventListener('click',()=>{menu.setAttribute('aria-expanded','false');$('nav').classList.remove('open');}));
  document.addEventListener('keydown',e=>{if(e.key==='Escape') {menu.setAttribute('aria-expanded','false');$('nav').classList.remove('open');}});
  let frame=0;
  window.addEventListener('scroll',()=>{if(frame)return;frame=requestAnimationFrame(()=>{const total=document.documentElement.scrollHeight-window.innerHeight;document.documentElement.style.setProperty('--progress',total?`${window.scrollY/total*100}%`:'0%');frame=0;});},{passive:true});
})();
