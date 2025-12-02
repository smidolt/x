# Preprocessing (modular)

Run all steps independently (each takes the original image, no chaining):

```bash
python -m src.main --image input/x.jpg --output output/check_steps_single --target-dpi 300
```

Example outputs:
- `output/check_steps_single/x__grayscale.png`
- `output/check_steps_single/x__crop_to_content.png`
- `output/check_steps_single/x__deskew.png` (stays straight on flat pages)

Sample deskew metadata:
```
deskew {'step': 'deskew', 'applied': False, 'warning': None, 'elapsed_seconds': 0.10, 'output_path': 'output/check_steps_single/x__deskew.png'}
```
