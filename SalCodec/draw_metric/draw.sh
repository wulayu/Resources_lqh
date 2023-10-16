# /bin/bash
set -e;
python draw_uvg.py >/dev/null &
python draw_mcl.py >/dev/null &
python draw_jct-vc-720p.py >/dev/null &
python draw_jct-vc-1080p.py >/dev/null &
echo drawing ...
wait
rm -rf result
mkdir result
cp ./*/*.png result/
zip -r result.zip result/*