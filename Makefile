.PHONY: help setup convert arms profile smoke pretrained scratch frozen-stem frozen-backbone transfer radiometry modality eval test lint clean

FLIR ?= Dataset/FLIR_ADAS_v2

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## install dependencies into .venv
	# Both extras together: uv sync prunes anything not named, so syncing one
	# extra alone silently uninstalls the other.
	uv sync --extra dev --extra wandb

convert:  ## build the thermal 8-bit arm from the FLIR download
	uv run thermaldet convert --arm agc --flir-root $(FLIR)

arms:  ## build all four arms (~2 min, ~6 GB)
	uv run thermaldet convert --arm agc    --flir-root $(FLIR)
	uv run thermaldet convert --arm global --flir-root $(FLIR)
	uv run thermaldet convert --arm p1p99  --flir-root $(FLIR)
	uv run thermaldet convert --arm rgb    --flir-root $(FLIR)

profile:  ## measure class balance, object scale and dynamic range -> reports/
	uv run thermaldet profile --flir-root $(FLIR)

smoke:  ## 1-epoch run to prove the pipeline works
	uv run thermaldet train smoke

pretrained:  ## the control: COCO-pretrained, fully fine-tuned
	uv run thermaldet train pretrained

scratch:  ## same architecture, random init
	uv run thermaldet train scratch

frozen-stem:  ## COCO-pretrained with layers 0-1 frozen
	uv run thermaldet train frozen_stem

frozen-backbone:  ## COCO-pretrained with the whole backbone frozen
	uv run thermaldet train frozen_backbone

transfer: pretrained scratch frozen-stem frozen-backbone  ## the whole transfer axis

radiometry:  ## the two 16-bit arms, against the 8-bit control
	uv run thermaldet train global_map
	uv run thermaldet train p1p99_map

modality:  ## the visible-spectrum arm, against the thermal control
	uv run thermaldet train rgb

eval:  ## score every finished run on the held-out test split -> reports/
	uv run thermaldet eval runs/train/*/weights/best.pt --split test

test:  ## run the test suite
	uv run pytest

lint:  ## lint and format check
	uv run ruff check src tests
	uv run ruff format --check src tests

clean:  ## remove generated runs and reports (not the built arms)
	rm -rf runs reports
