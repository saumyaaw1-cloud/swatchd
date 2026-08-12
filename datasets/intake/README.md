# Swatch'd Eyeliner Vision Dataset Intake

Drop source images into `datasets/intake/raw/<state>/`.

This intake is for collecting the next training set for state-aware eyeliner coaching. The first model only learned one class, `eyeliner`. The next model should learn what stage the liner is in and when the pen/hand is not the liner.

## Target Folders

- `00_no_liner`: Eye visible, no eyeliner mark. Include bare lids, lashes, shadows, and different lighting.
- `01_pen_near_eye`: Eyeliner pen or brush near the eye, but no new liner mark. Include the green circle pen.
- `02_root_mark`: The first small mark at the outer lash root.
- `03_tail_only`: Outer flick/tail drawn, not connected back yet.
- `04_return_line`: Tail plus return line, triangle outline visible.
- `05_filled_wing`: Wing filled in and connected to lash line.
- `06_too_long`: Wing extends too far past the intended guide.
- `07_too_low`: Wing droops downward or cuts into the eye shape.
- `08_too_thick`: Wing or lash line is too heavy for the intended look.
- `09_occluded_hand`: Fingers, brush, or pen blocking the eye area.

## Best Images To Collect First

Prioritize phone-camera images over polished stock photos. The app sees webcam/phone reality: motion blur, bathroom light, hands, shadows, pens, lashes, and uneven angles.

For each person/session, try to capture:

1. No liner, eye open.
2. No liner, eye half-closed.
3. Pen near eye, no mark.
4. First root mark.
5. Tail only.
6. Return line.
7. Filled wing.
8. One intentional mistake, such as too long or too low.

## Minimum Useful Counts

For the next model, aim for:

- 80+ `00_no_liner`
- 80+ `01_pen_near_eye`
- 80+ `03_tail_only`
- 80+ `04_return_line`
- 80+ `05_filled_wing`
- 40+ each mistake folder
- 40+ `09_occluded_hand`

That is enough to train a noticeably smarter prototype. A product-quality model will need much more variety.

## Licensing Notes

Use your own photos and volunteer photos whenever possible. For stock images, only use sources with clear reuse rights, and track the source in `sources.csv`.

Do not use random Pinterest, Instagram, TikTok, YouTube screenshots, or Google Images unless you have permission. Those are okay for visual inspiration, not for training data in a product.

## Labeling

Once enough images are in `raw/`, use Labelme and draw polygons around only the actual eyeliner mark. For negative folders like `00_no_liner`, `01_pen_near_eye`, and `09_occluded_hand`, keep images in the dataset even if there is no eyeliner polygon; they teach the model what not to detect.

## Importing Permitted URLs

If you have permission to use online images, add direct image URLs to `permitted_image_urls.csv`:

```csv
state,url,source_url,permission_notes
03_tail_only,https://example.com/image.jpg,https://example.com/page,Permission confirmed
```

Then run:

```sh
npm run dataset:intake:import
```

Use state folder names such as `03_tail_only`, or the short state name `tail_only`.

The importer downloads the file into `raw/<state>/` and appends the source to `sources.csv`.

## Preparing Images For LabelMe

Normalize everything in `raw/` and create blank LabelMe annotation files with:

```sh
npm run dataset:intake:prepare
```

The prepared JPEG and JSON pairs appear in `datasets/intake/labelme/`. State folder names are preserved in each filename. Open that folder in LabelMe, draw polygons labeled `eyeliner` around the actual liner, and leave negative-image JSON files with an empty `shapes` list.
