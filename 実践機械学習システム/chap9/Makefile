CHART_DIR = charts

sox sine_a.wav sine_b.wav sine_mix.wav:
	sox --null -r 22050 sine_a.wav synth 0.2 sine 400
	sox --null -r 22050 sine_b.wav synth 0.2 sine 3000
	sox --combine mix --volume 1 sine_b.wav --volume 0.5 sine_a.wav sine_mix.wav

fft_demo: sine_a.wav sine_b.wav sine_mix.wav
	python fft.py 
	convert sine_a_wav_fft.png sine_b_wav_fft.png sine_mix_wav_fft.png -append fft_demo.png
	
	python fft.py /media/sf_P/pymlbook-data/09-genre-class/genres/jazz/jazz.00012.wav
	mv jazz.00012_wav_fft.png fft_example.png
	

