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
		"torch>=1.12.1",
		"torchaudio>=0.12.1",
		"torchvision>=0.13.1",
		"torchmetrics>=0.9.3",
		"matplotlib>=3.5.3",
		"numpy>=1.23.2",
		"Pillow>=9.3.0",
		"scikit-learn>=1.1.3",
		"wandb>=0.13.5",
		"jupyterlab>=3.4.5",
		"pandas>=1.4.3",
		"tqdm>=4.64.0",
	],
)
