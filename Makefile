.PHONY: install test m0 m1 m2 m3 m4 m5 figures

install:
	pip install -e .

test:
	pytest -q

m0:
	python experiments/m0_smoke.py

m1:
	python experiments/m1_collision_audit.py --out results/m1_collisions

m2:
	python experiments/m2_tiny_train.py --config configs/m2_tiny.yaml

m3:
	python experiments/m3_matched_param.py --config configs/m3_50m_onehot.yaml
	python experiments/m3_matched_param.py --config configs/m3_50m_wave.yaml

m4:
	@for a in dense onehot wave hash albert; do \
	  for s in 0 1 2; do \
	    python experiments/m4_headline.py --config configs/m4_124m_$$a.yaml --seed $$s; \
	  done; \
	done

m5:
	python experiments/m5_long_tokens.py

figures:
	@echo "Regenerate every figure from results/. No hand-edited figures."
