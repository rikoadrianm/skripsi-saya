import Augmentor

p = Augmentor.Pipeline("Dataset/Unknow")

# rotasi kecil
p.rotate(probability=0.8, max_left_rotation=15, max_right_rotation=15)

# flip horizontal
p.flip_left_right(probability=0.5)

# flip vertical
p.flip_top_bottom(probability=0.3)

# zoom random
p.zoom_random(probability=0.6, percentage_area=0.7)

# perubahan brightness
p.random_brightness(probability=0.5, min_factor=0.6, max_factor=1.4)


p.sample(325)
