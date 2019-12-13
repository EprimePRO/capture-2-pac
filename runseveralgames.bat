for /l %%x in (21,1,51) do python capture2.py -i 600 -r 2-Pac-friendly -b 2-Pac-final -c -n 5 -q -s %%x --record ../recordings%%x >> batch.txt
pause