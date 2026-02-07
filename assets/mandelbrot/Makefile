.PHONY: default
default: mandelbrot.spv

mandelbrot.spv: mandelbrot.slang
	slangc -O3 -target spirv -o mandelbrot.spv mandelbrot.slang
