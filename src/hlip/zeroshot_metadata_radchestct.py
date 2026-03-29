CLASSNAMES = (
    "Emphysema", "Atelectasis", "Lung nodule",
    "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis", 
    "Interlobular septal thickening", "Cardiomegaly", "Pericardial effusion",
    "Hiatal hernia", "Calcification",
)


TEMPLATES = (
    lambda c: f"This study shows: there is {c}.",
    lambda c: f"This study shows: {c} is observed.",
    lambda c: f"This study shows: {c} is present.",
    lambda c: f"This study shows: onsistent with {c}.",
    lambda c: f"This study shows: compatible with {c}.",
)