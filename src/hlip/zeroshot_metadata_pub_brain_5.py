CLASSNAMES = [
    "Normal", "Stroke", "Glioma", "Meningioma", "Metastasis"
]

TEMPLATES = {
    "template": (lambda c: f'This MRI study shows: {c}.',),
}

PROMPTS = {
    "prompt": ("No significant abnormalities", "Acute stroke", "Glioma", "Meningioma", "Metastasis"),
}