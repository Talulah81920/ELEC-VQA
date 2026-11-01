// CPU-only interaction checks. This DOM fixture does not replace visual browser QA.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
const root=path.dirname(fileURLToPath(import.meta.url)),dist=path.join(root,'dist');
const html=fs.readFileSync(path.join(dist,'index.html'),'utf8');
const elements=new Map(),all=[];
class Element {
 constructor(tag='div'){this.tag=tag;this.children=[];this.dataset={};this.listeners={};this.attrs={};this.hidden=false;this.disabled=false;this.checked=false;this.classes=new Set();this.style={setProperty:(k,v)=>{this.attrs[k]=v}};this.classList={toggle:(k,on)=>{const v=on===undefined?!this.classes.has(k):on;if(v)this.classes.add(k);else this.classes.delete(k);return v},remove:k=>this.classes.delete(k)};}
 set textContent(v){this.children=[];this.text=String(v)}get textContent(){return (this.text||'')+this.children.map(x=>x.textContent).join('')}
 append(...xs){this.children.push(...xs)}prepend(...xs){this.children.unshift(...xs)}replaceChildren(...xs){this.text='';this.children=xs}
 setAttribute(k,v){this.attrs[k]=v}getAttribute(k){return this.attrs[k]}
 addEventListener(k,f){(this.listeners[k]??=[]).push(f)}async emit(k,extra={}){for(const f of this.listeners[k]||[])await f({target:this,key:'',preventDefault(){},...extra})}
 async click(){if(!this.disabled)await this.emit('click')}focus(){this.focused=true}showModal(){this.open=true}close(){this.open=false}
 get firstElementChild(){return this.children[0]}get lastElementChild(){return this.children.at(-1)}querySelectorAll(){return []}
}
for(const m of html.matchAll(/<([a-z][\w-]*)\b([^>]*?)>/gi)){
 const n=new Element(m[1]),attrs=m[2];
 for(const a of attrs.matchAll(/([\w-]+)="([^"]*)"/g)){n.attrs[a[1]]=a[2];if(a[1]==='id'){assert(!elements.has(a[2]),'Duplicate ID');elements.set(a[2],n)}if(a[1]==='class')n.classes=new Set(a[2].split(' '));if(a[1].startsWith('data-'))n.dataset[a[1].slice(5)]=a[2];}
 if(/\bchecked\b/.test(attrs))n.checked=true;all.push(n);
}
const match=(n,s)=>s.startsWith('.')?n.classes.has(s.slice(1)):s.startsWith('[data-')?s.slice(6,-1) in n.dataset:false;
const document={getElementById:id=>{assert(elements.has(id),'Missing actual element: '+id);return elements.get(id)},createElement:tag=>new Element(tag),createTextNode:text=>{const x=new Element('#text');x.textContent=text;return x},querySelectorAll:s=>all.filter(n=>match(n,s)),querySelector:s=>all.find(n=>match(n,s)),documentElement:new Element(),addEventListener(){}};
elements.get('evidence-toggle').append(new Element('span'),new Element('span'));
const sandbox={document,window:{addEventListener(){},scrollY:0,innerHeight:800},navigator:{clipboard:{writeText:async()=>{}}},requestAnimationFrame:f=>{f();return 1},console};
vm.createContext(sandbox);vm.runInContext(fs.readFileSync(path.join(dist,'data.js'),'utf8'),sandbox);vm.runInContext(fs.readFileSync(path.join(dist,'app.js'),'utf8'),sandbox);
const $=id=>document.getElementById(id),button=(attr,v)=>all.find(n=>n.dataset[attr]===v);
assert.equal($('example-list').children.length,8);assert.equal($('chart-bars').children.length,3);assert($('chart-result').textContent.includes('18.07'));
await button('track','direct').click();assert.equal($('chart-bars').children.length,5);assert($('chart-bars').textContent.includes('74.27'));assert(button('metric','joint').disabled);
$('supplemented').checked=false;await $('supplemented').emit('change');assert($('chart-bars').textContent.includes('74.18'));assert($('chart-bars').textContent.includes('52.37'));
await button('track','explain').click();await button('metric','joint').click();assert($('chart-bars').textContent.includes('63.76'));
$('supplemented').checked=true;await $('supplemented').emit('change');assert($('chart-bars').textContent.includes('64.24'));assert($('chart-bars').textContent.includes('46.17'));
await button('metric','answer').click();assert($('chart-bars').textContent.includes('70.16'));assert($('chart-bars').textContent.includes('67.97'));
await $('evidence-toggle').click();assert.equal($('example-evidence').hidden,false);assert($('example-evidence').textContent.includes('After-jumper route'));
await $('example-next').click();assert.equal($('example-evidence').hidden,true);assert.equal($('example-count').textContent,'2 / 8');
await $('example-prev').click();await $('example-prev').click();assert.equal($('example-count').textContent,'8 / 8');
await $('example-zoom').click();assert.equal($('image-dialog').open,true);await $('native-size').click();assert($('image-viewport').classes.has('native'));await $('close-dialog').click();assert.equal($('image-dialog').open,false);
await button('pair','state_invariance').click();assert($('pair-bars').textContent.includes('11.34%'));assert($('pair-bars').textContent.includes('61.86%'));
await button('pair','visual_flip').click();assert($('pair-bars').textContent.includes('4.35%'));assert($('pair-bars').textContent.includes('39.13%'));
await button('case','direct').click();assert.equal($('case-a').textContent,'No.');assert.equal($('case-e').textContent,'No.');
await button('case','explain').click();assert($('case-e').textContent.includes('Answer: Yes.'));
await $('target-a').click();assert(!$('target-content').textContent.includes('COMPACT EVIDENCE'));await $('target-e').click();assert($('target-content').textContent.includes('COMPACT EVIDENCE'));
for(const m of html.matchAll(/(?:href|src)="([^"]+)"/g)){
 const u=m[1];if(u.startsWith('https:')||u==='#')continue;
 if(u.startsWith('#'))assert(elements.has(u.slice(1)),'Missing anchor '+u);else assert(fs.existsSync(path.join(dist,u)),'Missing resource '+u);
}
console.log('PASS: protocols, scoring modes, original/supplemented values, examples, evidence, original-pixel viewer, case outputs, pair subsets, target tabs and local resources.');
console.log('Visual browser layout remains to be checked separately.');
