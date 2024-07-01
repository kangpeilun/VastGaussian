# render and eval rubble
#python render.py -s ../datasets/Mill19/rubble --exp_name rubble --eval --manhattan --resolution 4 --pos "25.607364654541 0.000000000000 -12.012700080872" --rot "0.923032462597 0.000000000000 0.384722054005 0.000000000000 1.000000000000 0.000000000000 -0.384722054005 0.000000000000 0.923032462597" --load_iteration 60_000
python render.py -s ../datasets/Mill19/rubble --exp_name rubble --eval --manhattan --plantform "tj" --resolution 4 --pos "0.0 0.0 0.0" --rot "0.0 21 0.0" --load_iteration 60_000
python metrics.py -m output/rubble

# render and eval checkpoint rubble
#python render.py -s ../datasets/Mill19/rubble --exp_name rubble --eval --load_iteration 60_000
#python metrics.py -m output/rubble

# render building
#python render.py -s ../datasets/Mill19/building --exp_name building --eval --manhattan --pos "-62.527942657471 0.000000000000 -15.786898612976" --rot "0.932374119759 0.000000000000 0.361494839191 0.000000000000 1.000000000000 0.000000000000 -0.361494839191 0.000000000000 0.932374119759" --load_iteration 60_000
#python metrics.py -m output/building