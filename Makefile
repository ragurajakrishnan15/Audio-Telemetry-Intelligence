.PHONY: install pipeline test dashboard clean

install:
	pip install -r requirements.txt

pipeline:
	python run_pipeline.py --step all

test:
	python -m pytest tests -q -rs

dashboard:
	streamlit run src/dashboard/app.py

clean:
	rm -rf data .pytest_cache
