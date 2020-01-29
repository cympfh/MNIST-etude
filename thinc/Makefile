TAG=thinc:latest

run: build
	docker run --rm -it $(TAG) mypy .
	docker run --rm -it $(TAG) python ./linear.py

build:
	docker build -t $(TAG) .
