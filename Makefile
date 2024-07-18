# Makefile

# Go parameters
GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
BINARY_NAME=golf

# OpenCV parameters
CXXFLAGS := $(shell pkg-config --cflags opencv4)
LDFLAGS := $(shell pkg-config --libs-only-L --libs-only-other opencv4)
LIBS := $(shell pkg-config --libs-only-l opencv4 | tr ' ' '\n' | sort -u | tr '\n' ' ')

all: build

build:
	CGO_CXXFLAGS=$(CXXFLAGS) CGO_LDFLAGS="$(LDFLAGS) $(LIBS)" $(GOBUILD) -o $(BINARY_NAME) *.go

clean:
	$(GOCLEAN)
	rm -f $(BINARY_NAME)

test:
	$(GOTEST) -v ./...
