.PHONY: setup verify prepare baseline train eval compare app api

setup:
	python -m pip install --upgrade pip
	python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
	python -m pip install -r requirements.txt

verify:
	python scripts/00_verify_gpu.py

prepare:
	python scripts/01_prepare_dataset.py

baseline:
	python scripts/02_baseline_eval.py

train:
	python scripts/03_train_qlora.py

eval:
	python scripts/04_evaluate_adapter.py

compare:
	python scripts/06_compare_results.py

app:
	streamlit run app/streamlit_app.py

api:
	uvicorn app.api:app --host 0.0.0.0 --port 8000
