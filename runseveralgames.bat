for /l %%x in (200,1,250) do python capture2.py -i 600 -r 2-Pac-friendly -b 2-Pac-final -c -n 3 -q -s %%x --record ../recordings%%x >> batch.txt
pause