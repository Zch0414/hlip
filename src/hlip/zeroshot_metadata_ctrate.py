CLASSNAMES = (
    "Emphysema", "Atelectasis", "Lung nodule", 
    "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion", 
    "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation", 
    "Bronchiectasis", "Interlobular septal thickening", "Cardiomegaly", 
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia", 
    "Arterial wall calcification",
)


TEMPLATES = (
    lambda c: f"This study shows: there is {c}.",
    lambda c: f"This study shows: {c} is observed.",
    lambda c: f"This study shows: {c} is present.",
    lambda c: f"This study shows: onsistent with {c}.",
    lambda c: f"This study shows: compatible with {c}.",
)
