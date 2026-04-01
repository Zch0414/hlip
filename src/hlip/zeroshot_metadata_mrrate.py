CLASSNAMES = (
    "Cerebral infarction", "Cerebral hemorrhage", "Metastatic malignant neoplasm to brain", "Intracranial meningioma",
    "Demyelinating disease of central nervous system", "Herniation of nucleus pulposus", "Spinal cord compression", 
    "Lacunar infarct", "Silent micro-hemorrhage of brain", "Cavernous hemangioma", "Subdural intracranial hemorrhage",
    "Gliosis", "Cerebral atrophy", "Encephalomalacia", "Arachnoid cyst", "Empty sella syndrome", "Intracranial aneurysm",
    "Chiari malformation", "Schwannoma", "Cyst of pineal gland", "Hemangioma of vertebral column", "Rathke's pouch cyst", 
    "Cerebral edema", "Spinal stenosis", "Mastoiditis", "Chronic mastoiditis", "Ventriculomegaly", "Cerebellar degeneration", 
    "Mega cisterna magna", "Structure of cave of septum pellucidum", "Hyperostosis of skull", "Watershed infarct", "Choroid plexus cyst",
    "Foraminal Spinal Stenosis", "Lipoma of brain", "Glioma", "Pituitary adenoma"
)


TEMPLATES = (
    lambda c: f"This study shows: {c}.",
    lambda c: f"This study shows: {c} identified.", 
    lambda c: f"This study shows: {c} noted.", 
    lambda c: f"This study shows: {c} seen.", 
    lambda c: f"This study shows: new {c}.",
    lambda c: f"This study shows: known {c}.",
    lambda c: f"This study shows: prominent {c}.",
    lambda c: f"This study shows: likely {c}.",
    lambda c: f"This study shows: possibly {c}.",
    lambda c: f"This study shows: indicating {c}.",
    lambda c: f"This study shows: reflecting {c}.",
    lambda c: f"This study shows: representing {c}.",
    lambda c: f"This study shows: suggesting {c}.",
    lambda c: f"This study shows: indicative of {c}.",
    lambda c: f"This study shows: suggestive of {c}.",
    lambda c: f"This study shows: related to {c}.",
    lambda c: f"This study shows: consistent with {c}.",
    lambda c: f"This study shows: compatible with {c}.",
)