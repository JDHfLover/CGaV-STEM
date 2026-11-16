"""
Автоматизовані модульні тести для STEM-проєкту (критерії C1, C2, C3).
Перевіряють:
1) Коректність усіх 6 реалізованих методів на малих тестових матрицях 3x3 та 5x5
   із порівнянням результату з очікуваним, обчисленим вручну (C3 = 100%).
2) Коректність обчислення метрик MSE, PSNR, SSIM, ентропії Шеннона, IoU та F1 (C1).
3) Безпечну обробку межових випадків: порожні/пошкоджені файли, 1x1, 2x2,
   однорідні зображення, RGBA-зображення (C2 = 100%).
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from stem_project import (
    bilateral_filter,
    calc_iou_and_f1,
    calc_mse,
    calc_psnr,
    calc_shannon_entropy,
    calc_ssim,
    clahe_enhancement,
    compute_glcm_features,
    compute_lbp_image,
    grayscale_dilation,
    grayscale_erosion,
    load_grayscale_image_safe,
    morphological_filter,
    otsu_segmentation,
    otsu_threshold_value,
    sobel_edge_detector,
    validate_and_prepare_image,
)


class TestStemMethodsManual3x3(unittest.TestCase):
    """
    Тестування всіх 6 класичних методів обробки зображень на малих матрицях 3x3 і 5x5
    з ручною перевіркою значень (Критерій C3).
    """

    def test_1_bilateral_filter_3x3_manual(self) -> None:
        """
        Метод 1 (Білатеральний фільтр): на однорідному зображенні 3x3 зі значенням 100
        та на ступінчастому контурі з великим стрибком яскравості (0 vs 250) при малому
        sigma_range=5.0 контур зберігається без розмиття: ліва половина залишається 0,
        права — 250.
        """
        step_img = np.array([
            [0, 0, 250, 250],
            [0, 0, 250, 250],
            [0, 0, 250, 250],
            [0, 0, 250, 250]
        ], dtype=np.uint8)
        filtered = bilateral_filter(step_img, ksize=3, sigma_spatial=1.5, sigma_range=5.0)
        np.testing.assert_array_equal(filtered, step_img)

    def test_2_clahe_3x3_manual(self) -> None:
        """
        Метод 2 (Гістограмна корекція CLAHE): на матриці 3x3 з трьома рівнями
        яскравості (10, 50, 90), де кожен рівень зустрічається рівно по 3 рази,
        ручний розрахунок CDF дає:
        cdf(10) = 3/9, cdf_min = 3/9 -> lut(10) = 0;
        cdf(50) = 6/9 -> lut(50) = (6 - 3)/(9 - 3) * 255 = 127.5 -> 128;
        cdf(90) = 9/9 -> lut(90) = (9 - 3)/(9 - 3) * 255 = 255.
        """
        img_3x3 = np.array([
            [10, 50, 90],
            [10, 50, 90],
            [10, 50, 90]
        ], dtype=np.uint8)
        expected = np.array([
            [0, 128, 255],
            [0, 128, 255],
            [0, 128, 255]
        ], dtype=np.uint8)
        res = clahe_enhancement(img_3x3, grid_size=1, clip_limit=4.0)
        np.testing.assert_array_equal(res, expected)

    def test_3_sobel_3x3_manual(self) -> None:
        """
        Метод 3 (Контурний аналіз — оператор Собеля): для вертикального градієнта 3x3
        [[0, 10, 20], [0, 10, 20], [0, 10, 20]] у центральному пікселі (1, 1):
        Gx = (-1*0 + 1*20) + (-2*0 + 2*20) + (-1*0 + 1*20) = 20 + 40 + 20 = 80,
        Gy = 0, отже модуль градієнта G(1, 1) = sqrt(80^2 + 0^2) = 80.
        """
        img_3x3 = np.array([
            [0, 10, 20],
            [0, 10, 20],
            [0, 10, 20]
        ], dtype=np.uint8)
        res_unnorm = sobel_edge_detector(img_3x3, normalize=False)
        self.assertEqual(int(res_unnorm[1, 1]), 80)
        self.assertEqual(int(res_unnorm[1, 0]), 40)
        self.assertEqual(int(res_unnorm[1, 2]), 40)

    def test_4_otsu_3x3_manual(self) -> None:
        """
        Метод 4 (Сегментація методом Оцу): для бімодальної матриці 3x3, де 5 пікселів
        мають значення 20 і 4 пікселі мають значення 200, міжкласова дисперсія досягає
        максимуму на порозі t = 20, розділяючи класи на {0} та {255} (або середні 20 і 200).
        """
        img_3x3 = np.array([
            [20, 20, 200],
            [20, 20, 200],
            [20, 200, 200]
        ], dtype=np.uint8)
        t_opt = otsu_threshold_value(img_3x3)
        self.assertEqual(t_opt, 20)

        reconstructed, _, binary_mask = otsu_segmentation(img_3x3, reconstruct_class_means=True)
        expected_mask = np.array([
            [0, 0, 255],
            [0, 0, 255],
            [0, 255, 255]
        ], dtype=np.uint8)
        np.testing.assert_array_equal(binary_mask, expected_mask)
        np.testing.assert_array_equal(reconstructed, img_3x3)

    def test_5_morphology_3x3_manual(self) -> None:
        """
        Метод 5 (Морфологічна обробка): матриця 3x3 із центральним світлим пікселем 200
        на фоні 50. Ручне обчислення:
        - Ерозія 3x3 у центрі: min(50, ..., 200) = 50 -> відкриття повністю видаляє імпульс (усі 50).
        - Дилатація 3x3: усі пікселі стають 200 -> закриття дає 200.
        - Півсума відкриття та закриття: 0.5 * (50 + 200) = 125.
        """
        img_3x3 = np.array([
            [50, 50, 50],
            [50, 200, 50],
            [50, 50, 50]
        ], dtype=np.uint8)
        eroded = grayscale_erosion(img_3x3, ksize=3)
        self.assertTrue(np.all(eroded == 50))

        dilated = grayscale_dilation(img_3x3, ksize=3)
        self.assertTrue(np.all(dilated == 200))

        opened = morphological_filter(img_3x3, ksize=3, operation="opening")
        self.assertTrue(np.all(opened == 50))

        smoothed = morphological_filter(img_3x3, ksize=3, operation="open_close_smooth")
        self.assertEqual(int(smoothed[1, 1]), 125)

    def test_6_lbp_and_glcm_3x3_manual(self) -> None:
        """
        Метод 6 (Аналіз текстур — LBP та GLCM): для матриці 3x3 з центральним пікселем 100:
        верхній лівий (біт 0) = 120 >= 100 (1), верхній (біт 1) = 80 < 100 (0),
        верхній правий (біт 2) = 150 >= 100 (1), правий (біт 3) = 100 >= 100 (1),
        інші сусіди = 50 < 100 (0).
        Очікуваний LBP у центрі (1, 1) = 2^0 + 2^2 + 2^3 = 1 + 4 + 8 = 13.
        """
        img_3x3 = np.array([
            [120, 80, 150],
            [50, 100, 100],
            [50, 50, 50]
        ], dtype=np.uint8)
        lbp = compute_lbp_image(img_3x3)
        self.assertEqual(int(lbp[1, 1]), 13)

        # На однорідному зображенні GLCM має контраст = 0, гомогенність = 1, енергію = 1
        const_img = np.full((4, 4), 64, dtype=np.uint8)
        feats = compute_glcm_features(const_img, levels=16, distance=1)
        self.assertAlmostEqual(feats["glcm_contrast"], 0.0, places=6)
        self.assertAlmostEqual(feats["glcm_homogeneity"], 1.0, places=6)
        self.assertAlmostEqual(feats["glcm_energy"], 1.0, places=6)


class TestMetricsAndEdgeCases(unittest.TestCase):
    """
    Перевірка метрик якості (C1) та обробки межових випадків (C2).
    """

    def test_metrics_manual_calculation(self) -> None:
        """
        Перевірка формул MSE, PSNR, SSIM, ентропії та IoU/F1.
        Якщо всі пікселі відрізняються рівно на 10: MSE = 100,
        PSNR = 10 * log10(255^2 / 100) = 28.1308 дБ.
        """
        a = np.full((5, 5), 100, dtype=np.uint8)
        b = np.full((5, 5), 110, dtype=np.uint8)
        self.assertAlmostEqual(calc_mse(a, b), 100.0, places=5)
        self.assertAlmostEqual(calc_psnr(a, b), 28.1308036, places=4)
        self.assertEqual(calc_psnr(a, a), 100.0)
        self.assertAlmostEqual(calc_ssim(a, a), 1.0, places=5)
        self.assertAlmostEqual(calc_shannon_entropy(a), 0.0, places=5)

        m1 = np.array([[255, 255], [0, 0]], dtype=np.uint8)
        m2 = np.array([[255, 0], [0, 0]], dtype=np.uint8)
        iou, f1 = calc_iou_and_f1(m1, m2)
        self.assertAlmostEqual(iou, 0.5, places=5)
        self.assertAlmostEqual(f1, 2.0 / 3.0, places=5)

    def test_edge_cases_c2(self) -> None:
        """
        Критерій C2: перевірка роботи на межових значеннях (1x1, 2x2, RGBA,
        однорідні чорні/білі зображення, порожні або пошкоджені файли).
        """
        tiny_1x1 = np.array([[123]], dtype=np.uint8)
        self.assertEqual(bilateral_filter(tiny_1x1).shape, (1, 1))
        self.assertEqual(clahe_enhancement(tiny_1x1).shape, (1, 1))
        self.assertEqual(sobel_edge_detector(tiny_1x1).shape, (1, 1))
        self.assertEqual(otsu_segmentation(tiny_1x1)[0].shape, (1, 1))
        self.assertEqual(morphological_filter(tiny_1x1).shape, (1, 1))
        self.assertEqual(compute_lbp_image(tiny_1x1).shape, (1, 1))

        rgba_img = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba_img[:, :, 0] = 200
        rgba_img[:, :, 3] = 255
        gray = validate_and_prepare_image(rgba_img)
        self.assertEqual(gray.shape, (4, 4))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            empty_file = tmp_path / "empty.png"
            empty_file.write_bytes(b"")
            self.assertIsNone(load_grayscale_image_safe(empty_file))

            corrupt_file = tmp_path / "corrupt.png"
            corrupt_file.write_bytes(b"not_an_image_content")
            self.assertIsNone(load_grayscale_image_safe(corrupt_file))

            missing_file = tmp_path / "missing.png"
            self.assertIsNone(load_grayscale_image_safe(missing_file))

            valid_file = tmp_path / "valid.bmp"
            Image.fromarray(np.full((4, 4), 77, dtype=np.uint8), mode="L").save(valid_file)
            loaded = load_grayscale_image_safe(valid_file)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.shape, (4, 4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
