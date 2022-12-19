import setuptools

setuptools.setup(
	name="playing_cards_concept_models",
	version="0.0.1",
	author="Jack Furby",
	author_email="furbyjl@cardiff.ac.uk",
	description="Concept Models",
	url="https://github.com/JackFurby/Playing-cards-CBM",
	packages=setuptools.find_packages(),
	python_requires='>=3.8, <=3.11',
	install_requires=[
		"torch>=1.13.1",
		"torchaudio>=0.13.1",
		"torchvision>=0.14.1",
		"torchmetrics>=0.11.0",
		"matplotlib>=3.6.2",
		"numpy>=1.24.0",
		"Pillow>=9.3.0",
		"scikit-learn>=1.2.0",
		"wandb>=0.13.7",
		"jupyterlab>=3.5.1",
		"pandas>=1.5.2",
		"tqdm>=4.64.0",
		"opencv-python>=4.6.0.66",
	],
)
