CLASSNAMES = [
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Arterial wall calcification",
]

ORGANS = {
    "Emphysema": "lung",
    "Atelectasis": "lung",
    "Lung nodule": "lung",
    "Lung opacity": "lung",
    "Pulmonary fibrotic sequela": "lung",
    "Pleural effusion": "lung",
    "Peribronchial thickening": "lung",
    "Consolidation": "lung",
    "Bronchiectasis": "lung",
    "Interlobular septal thickening": "lung",
    "Cardiomegaly": "heart",
    "Pericardial effusion": "heart",
    "Coronary artery wall calcification": "heart",
    "Hiatal hernia": "esophagus",
    "Arterial wall calcification": "aorta",
}

TEMPLATES = {
    "lung": (lambda c: f'The lung shows: {c}.',),
    "heart": (lambda c: f'The heart shows: {c}.',),
    "esophagus": (lambda c: f'The esophagus shows: {c}.',),
    "aorta": (lambda c: f'The aorta shows: {c}.',),
    "volume": (lambda c: f'The volume shows: {c}.',),
}


PROMPTS = {
    "Emphysema": ("Not emphysema", "Emphysema"),
    "Atelectasis": ("Not atelectatic", "Atelectasis"),
    "Lung nodule": ("Not nodule", "Nodule"),
    "Lung opacity": ("Not opacity", "Opacity"),
    "Pulmonary fibrotic sequela": ("Not pulmonary fibrotic sequela", "Pulmonary fibrotic sequela"),
    "Pleural effusion": ("Not pleural effusion", "Pleural effusion"),
    "Peribronchial thickening": ("Not peribronchial thickening", "Peribronchial thickening"),
    "Consolidation": ("Not consolidation", "Consolidation"),
    "Bronchiectasis": ("Not bronchiectasis", "Bronchiectasis"),
    "Interlobular septal thickening": ("Not interlobular septal thickening", "Interlobular septal thickening"),
    "Cardiomegaly": ("Not cardiomegaly", "Cardiomegaly"),
    "Pericardial effusion": ("Not pericardial effusion", "Pericardial effusion"),
    "Coronary artery wall calcification": ("Not coronary artery wall calcification", "Coronary artery wall calcification"),
    "Hiatal hernia": ("Not hiatal hernia", "Hiatal hernia"),
    "Arterial wall calcification": ("Not arterial wall calcification", "Arterial wall calcification"),
}