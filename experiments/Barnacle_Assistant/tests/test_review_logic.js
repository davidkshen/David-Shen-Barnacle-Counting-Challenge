// Run with `node tests/test_review_logic.js`. No external JS packages required.
// This tests actual inline reviewer logic through a DOM/canvas shim, not rendering.
const fs = require('fs');
const vm = require('vm');
const path = require('path');
const template = fs.readFileSync(path.join(__dirname, '../barnacles/review_template.html'), 'utf8');
const data = {
  name: 'Synthetic logic check', key: 'test', size: 1024, grid: 8,
  predictions: [[64, 64, 5, 1, .7, .02]], model_counts: [1, ...Array(63).fill(0)],
  random_sample: Array.from({length: 16}, (_, i) => i)
};
const setup = `
const noop = () => {};
const ctx = new Proxy({}, {get: () => noop, set: () => true});
const elements = new Map();
const document = {getElementById(id) {
  if (!elements.has(id)) elements.set(id, {checked: true, value: 0,
    textContent: '', innerHTML: '', getContext: () => ctx,
    getBoundingClientRect: () => ({left: 0, top: 0, width: 600, height: 600})});
  return elements.get(id);
}, createElement: () => ({click: noop})};
const localStorage = {getItem: () => null, setItem: noop};
class Image {set src(v) {this.onload();}}
const confirm = () => true;
`;
const checks = `
function check(condition, message) {if (!condition) throw new Error(message);}
check(loaded, 'Image onload did not trigger drawing');
check(counts()[0] === 1, 'Initial count incorrect');
setTile(0);
zoom.onclick({clientX: 300, clientY: 300});
check(counts()[0] === 0, 'Remove click failed');
zoom.onclick({clientX: 300, clientY: 300});
check(counts()[0] === 1, 'Add click failed');
zoom.onclick({clientX: 0, clientY: 0});
check(counts()[0] === 1, 'A halo click must not edit the tile core');
for (let i = 0; i < 16; i++) {setTile(i); elements.get('complete').onclick();}
const payload = exportPayload();
check(payload.audit.complete && payload.audit.estimate === 1, 'Difference estimator incorrect');
check(payload.manual_sample_audit.estimate === 4, 'Manual estimator incorrect');
setTile(0); zoom.onclick({clientX: 300, clientY: 300});
check(!audit().complete, 'Editing must reopen a completed tile');
zoom.onclick({clientX: 300, clientY: 300});
for (let i = 0; i < 64; i++) {setTile(i); elements.get('complete').onclick();}
check(audit().estimate === 1 && audit().standard_error === 0, 'Census behavior incorrect');
`;
const script = template.split('<script>')[1].split('</script>')[0].replace('__DATA__', JSON.stringify(data));
vm.runInNewContext(setup + script + checks, {});
console.log('Reviewer logic passed: drawing trigger, add/remove, halo ownership, completion, both estimators, reopen-on-edit, census.');
