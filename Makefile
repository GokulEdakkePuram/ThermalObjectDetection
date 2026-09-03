.PHONY: help setup convert arms profile smoke pretrained scratch frozen radiometry eval test lint clean

FLIR ?= Dataset/FLIR_ADAS_1_3
# Set when the download's annotation JSONs are missing and YOLO labels for it
# already exist: make convert ADOPT="--adopt-labels $(FLIR)/yolo/labels"
ADOPT ?=

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## install dependencies into .venv
	# Both extras together: uv sync prunes anything not named, so syncing one
	# extra alone silently uninstalls the other.
	uv sync --extra dev --extra wandb

convert:  ## build the AGC arm from the FLIR download
	uv run thermaldet convert --arm agc --flir-root $(FLIR) $(ADOPT)

arms:  ## build all three preprocessing arms (~1 min, ~3.5 GB)
	uv run thermaldet convert --arm agc    --flir-root $(FLIR) $(ADOPT)
	uv run thermaldet convert --arm global --flir-root $(FLIR) $(ADOPT)
	uv run thermaldet convert --arm p1p99  --flir-root $(FLIR) $(ADOPT)

profile:  ## measure class balance, object scale and dynamic range -> reports/
	uv run thermaldet profile --flir-root $(FLIR)

smoke:  ## 1-epoch run to prove the pipeline works
	uv run thermaldet train smoke

pretrained:  ## the control: COCO-pretrained, fully fine-tuned
	uv run thermaldet train pretrained

scratch:  ## same architecture, random init
	uv run thermaldet train scratch

frozen:  ## COCO-pretrained with the backbone frozen
	uv run thermaldet train frozen_backbone

radiometry:  ## the two 16-bit arms, against the AGC control
	uv run thermaldet train global_map
	uv run thermaldet train p1p99_map

eval:  ## compare every finished run -> reports/results.md
	uv run thermaldet eval runs/train/*/weights/best.pt

test:  ## run the test suite
	uv run pytest

lint:  ## lint and format check
	uv run ruff check src tests
	uv run ruff format --check src tests

clean:  ## remove generated runs and reports (not the built arms)
	rm -rf runs reports
