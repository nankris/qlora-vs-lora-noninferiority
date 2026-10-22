from pathlib import Path
import time,json
r=Path('/content/qlora_vs_lora')
assert (r/'pilot.done').exists()
assert not (r/'confirm.start').exists()
(r/'confirm.start').write_text('a3a5d235b16cd8170635fd0c96d982c45648b70b')
(r/'confirmation_start.json').write_text(json.dumps({'unix':time.time(),'preregistration_commit':'a3a5d235b16cd8170635fd0c96d982c45648b70b'}))
print('Confirmation started under the committed protocol')
