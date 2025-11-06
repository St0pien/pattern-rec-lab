class DinoAugmentation:
    def __init__(
        self,
        base_transforms,
        student_augs,
        final_transforms,
        n_global_views=2,
        n_student_crops=5,
    ):
        self.base_transforms = base_transforms
        self.student_augs = student_augs
        self.final_transforms = final_transforms
        self.n_global_views = n_global_views
        self.n_crops = n_student_crops

    def __call__(self, x):
        x_views = [
            self.base_transforms(x) for _ in range(self.n_global_views + self.n_crops)
        ]
        for i in range(self.n_global_views, self.n_global_views + self.n_crops):
            x_views[i] = self.student_augs(x_views[i])
        return [self.final_transforms(view) for view in x_views]
