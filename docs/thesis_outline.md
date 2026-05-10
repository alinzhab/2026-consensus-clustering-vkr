# Структура текста ВКР

Рекомендуемая структура пояснительной записки, согласованная с тем, что
реально делает код. Каждый раздел содержит конкретные ссылки на файлы
проекта и таблицы, которые туда нужно перенести.

## Введение

- Актуальность: ансамблевые методы в кластеризации, проблема разнородности
  базовых разбиений, практическая значимость консенсуса.
- Цель работы: разработать и сравнить варианты консенсус-кластеризации,
  предложить модификации, оценить их экспериментально.
- Задачи: 4–6 штук, по числу основных компонент проекта (см. ниже).
- Объект и предмет исследования.
- Научная новизна — в `docs/contribution.md`.
- Положения, выносимые на защиту — там же.
- Практическая значимость: веб-стенд + переиспользуемое ядро.
- Структура работы.

## Глава 1. Аналитический обзор

- Постановка задачи консенсус-кластеризации, формальные определения.
- Co-association matrix, классическая иерархическая консенсус-схема
  (Fred & Jain 2005).
- Взвешенные ансамбли (Vega-Pons & Ruiz-Shulcloper 2011).
- SDGCA: similarity and dissimilarity guided matrix construction.
- Графовая диффузия как обобщение случайных блужданий.
- Метрики качества: NMI, ARI, pairwise F-score.
- Принципы статистического сравнения нескольких алгоритмов
  (Demsar 2006): Фридман + Неменьи, Вилкоксон + Холм.

## Глава 2. Реализация алгоритмов и архитектура системы

- Архитектура: ядро `consensus_lab/`, веб-приложение `app.py`,
  пакетные сценарии, реестр алгоритмов `algorithms_base.py`.
- Базовые кластеризации: `consensus_lab/base_clusterings.py` —
  k-means, иерархические, feature subsampling, шум.
- 4 алгоритма с краткими блок-схемами и листингами ключевых функций:
  - `hierarchical_consensus.py` — `build_coassociation_matrix`,
    `get_cls_result`.
  - `hierarchical_consensus_modified.py` — `compute_base_clustering_weights`,
    `build_weighted_consensus_matrix`.
  - `sdgca.py` — `compute_neci`, `compute_nwca`, `optimize_sdgca`.
  - `sdgca_modified.py` — `compute_modified_neci`,
    `graph_diffusion_of_cluster`.
- Генераторы синтетических данных.
- Веб-интерфейс: маршруты, схема БД, формат хранимых результатов.
- Развёртывание (`render.yaml`).

## Глава 3. Экспериментальное исследование

- Описание датасетов (`results/dataset_descriptions_detailed.md`).
- Протокол: `seed=19`, `m`, `runs`, перебор linkage.
- Полные таблицы: перенести из `results/analysis_full_suite.tsv` и
  `results/analysis_report.md`.
- Графики (если есть `generate_sdgca_analysis_plots.py`):
  - средние ранги по алгоритмам;
  - boxplot NMI по датасетам;
  - тепловые карты сравнения linkage.
- Статистическая проверка: запустить
  `python consensus_lab/statistical_tests.py --metric nmi_mean --baseline sdgca_modified`
  и перенести вывод (Фридман, Неменьи, Вилкоксон).
- Анализ: где модификации работают, где нет, почему.

## Глава 4. Веб-приложение

- Сценарии использования.
- Интерфейс: скриншоты главной, datasets, generate, test, results.
- Архитектура слоёв БД и обработчиков.
- Замечания по производительности и ограничения бесплатного хостинга.

## Заключение

- Краткие итоги по каждой задаче.
- Обоснованное соответствие положениям, выносимым на защиту.
- Перечень полученных артефактов: 4 алгоритма, бенчмарк, веб-стенд,
  модуль статистических тестов, набор синтетических датасетов.
- Направления развития — `docs/limitations.md`.

## Список литературы

- Fred A.L.N., Jain A.K. Combining Multiple Clusterings Using Evidence
  Accumulation, IEEE TPAMI 2005.
- Strehl A., Ghosh J. Cluster Ensembles — A Knowledge Reuse Framework
  for Combining Multiple Partitions, JMLR 2002.
- Vega-Pons S., Ruiz-Shulcloper J. A Survey of Clustering Ensemble
  Algorithms, IJPRAI 2011.
- Tao Z. et al. Similarity and Dissimilarity Guided Co-association
  matrix construction.
- Demsar J. Statistical Comparisons of Classifiers over Multiple
  Data Sets, JMLR 2006.
- Belkin M., Niyogi P. Laplacian Eigenmaps for Dimensionality Reduction
  and Data Representation, Neural Computation 2003.
- (RepliClust, DENSIRED, scikit-learn, SciPy, NumPy, Flask — для
  раздела «Используемые средства».)

## Приложения

- Приложение А: листинги ключевых функций (`build_coassociation_matrix`,
  `optimize_sdgca`, `graph_diffusion_of_cluster`,
  `compute_base_clustering_weights`).
- Приложение Б: полные таблицы метрик
  (`results/analysis_full_suite.tsv`).
- Приложение В: вывод `statistical_tests.py`.
- Приложение Г: руководство пользователя веб-приложения.
- Приложение Д: акт о внедрении / справка о практическом использовании
  (если применимо).
