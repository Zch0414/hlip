CLASSNAMES = [
    "Any", "Intraparenchymal", "Intraventricular", "Subarachnoid", "Subdural"
]


TEMPLATES = {
    "template": (lambda c: f"This CT study shows: {c}.",),
}


PROMPTS = {
    "prompt": ("intracranial hemorrhage", "intraparenchymal hemorrhage", "intraventricular hemorrhage", "subarachnoid hemorrhage", "subdural hemorrhage"),
}