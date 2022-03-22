# ADL2022 HW1

### Environment
`pip install -r requirements.txt`

### Download Model
`bash download.sh`

### Q1 Intent Classfication

$1 test data path e.g. ./data/intent/test.json

$2 csv output path e.g. ./intent.csv

`bash intent_cls.sh {$1} {$2}`

### Q2 Slot Tagging

$1 test data path e.g. ./data/slot/test.json

$2 csv output path e.g. ./slot.csv

`bash slot_tag.sh {$1} {$2}`


### Training Reproduce

`python3 ./train_intent.py`

`python3 ./train_slot.py`
