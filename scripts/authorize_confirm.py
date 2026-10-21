from pathlib import Path
import time,json
r=Path('/content/exp101c')
assert (r/'pilot.done').exists()
assert not (r/'confirm.authorized').exists()
(r/'confirm.authorized').write_text('a3a5d235b16cd8170635fd0c96d982c45648b70b')
(r/'confirmation_start.json').write_text(json.dumps({'unix':time.time(),'preregistration_commit':'a3a5d235b16cd8170635fd0c96d982c45648b70b'}))
print('Confirmation authorized by committed protocol')
