"""Fetch an official Mesa archive in bounded parallel byte ranges."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

url = 'https://archive.mesa3d.org/mesa-26.2.2.tar.xz'
root = Path('/home/nmt/isaaclab-vulkan-build')
size = 68533264
count = 8
chunk = (size + count - 1) // count

def fetch(i):
    start, end = i * chunk, min(size, (i + 1) * chunk) - 1
    request = Request(url, headers={'Range': f'bytes={start}-{end}'})
    with urlopen(request, timeout=90) as response:
        assert response.status == 206, response.status
        data = response.read()
    assert len(data) == end - start + 1
    (root / f'mesa.part{i}').write_bytes(data)
    print(f'Part {i + 1}/{count} downloaded', flush=True)

with ThreadPoolExecutor(max_workers=count) as pool:
    list(pool.map(fetch, range(count)))
with (root / 'mesa-26.2.2-parallel.tar.xz').open('wb') as target:
    for i in range(count):
        target.write((root / f'mesa.part{i}').read_bytes())
print('Archive assembled', flush=True)
