# Human activity recognition with a Hidden Markov Model

Formative 2. We record accelerometer and gyroscope data on our phones while standing, walking, jumping and leaving the phone still on a table, then infer the activity from the sensor signals with a Gaussian-emission HMM written from scratch in numpy (forward–backward, Baum–Welch, Viterbi).

```
HMM_activity_recognition.ipynb   the whole pipeline; Run All after the data is in place
data/raw/<member>/               phone exports, one folder per group member
data/segment_labels.csv          start/end times for the mixed-activity recordings
data/clean/<member>/             written by the notebook: one harmonised CSV per recording
figures/                         written by the notebook: every figure for the report
results/                         written by the notebook: every table for the report
requirements.txt
```

## Results (this submission)

The dataset is **65 labelled recordings** from a single contributor, collected on **two phones** and harmonised to a common 50 Hz: a Samsung SM-G973F logging a true **100 Hz** (the 63 single-activity training clips) and a Google Pixel 6 delivering **~55 Hz** (the two unseen ~45 s test recordings). Every activity clears the 90 s minimum.

A from-scratch Gaussian-emission HMM (forward–backward, Baum–Welch, Viterbi) reaches **81.4 % accuracy** on the two unseen Pixel 6 recordings — different device, different session from all training data.

| Activity | Sensitivity | Specificity |
|---|---|---|
| walking | 1.00 | 0.92 |
| still | 0.95 | 1.00 |
| standing | 0.73 | 0.85 |
| jumping | 0.44 | 1.00 |

- Baum–Welch converged in **5 iterations** (train log-likelihood −75.8 → −66.8, stopping threshold `BW_TOL`).
- The temporal model beats classifying each window independently: **75.7 % → 81.4 %**, driven by the Baum–Welch-refined transition matrix.
- Per test file: test 1 = **91.7 %**, test 2 = **70.6 %**. Jumping is the hardest activity (confused with standing at the steady start/end of each jump burst, blunted by the Pixel's lower sampling rate).

**Data-collection videos** (whole body + face, holding the phone while recording):

| Activity | Link |
|---|---|
| Standing | https://youtube.com/shorts/MuexVpJiZWw |
| Walking | https://youtube.com/shorts/fzsP2rYffkM |
| Jumping | https://youtube.com/shorts/NbohnmOFKUk |
| Still | https://youtube.com/shorts/TWcEYuwgiLQ |

## 1. Recording the data

**Phone placement.** Everyone holds the phone the same way, because orientation is one of the features. For standing, walking and jumping: phone in the right hand, upright (top edge pointing up), screen facing your body, at waist level. For still: phone flat on a table, screen up, nobody touching the table.

**App settings.** In Sensor Logger turn on Accelerometer, Gyroscope and Gravity, set the sampling rate to 100 Hz (or the highest your phone offers) and export each recording as a zipped CSV. If you use Physics Toolbox Sensor Suite instead, use Multi Record with the g-Force meter and the gyroscope selected. Note your phone model; it goes in the report next to your sampling rate.

**How much.** The rubric asks for about 50 clean files in total and at least 90 seconds of every activity. Record each clip for about 10 seconds; the notebook drops the first and last half second, where tapping start and stop shakes the phone. With three members, five clips of each activity per person gives 60 files and roughly 135 seconds per activity. With two members, record seven clips of each activity per person.

**Naming.** The file name is the label, so it has to start with the activity:

```
data/raw/member_a/walking_01.zip        Sensor Logger export, as downloaded
data/raw/member_a/still_03/             the same export, unzipped, also works
data/raw/member_b/jumping_02.csv        a single CSV with time, acc x/y/z and gyro x/y/z columns
```

Activity prefixes: `standing`, `walking`, `jumping`, `still`. Replace `member_a` with your first name.

**Unseen test recordings (at least two).** Record two continuous recordings of about one minute in which you go through all four activities, for example: phone on the table 10 s, pick it up, stand 8 s, walk 12 s, stand 5 s, jump 8 s, stand 5 s, put it down, still 10 s. Make them different from the training data on purpose: another member, another room or outdoors, another day. Name them `test_01_<member>.zip`, `test_02_<member>.zip`. They are never used for training.

**Optional training sessions.** One or two more continuous recordings named `session_01_<member>.zip` give the model real switches between activities to learn from. Without them, every training sequence contains a single activity and the transition matrix can only learn how long activities last, not which activity tends to follow which.

**Labelling the continuous recordings.** Film them. Start the video first, then tap record in the app in view of the camera; that frame is second 0 of the sensor recording. Write one row per activity segment into `data/segment_labels.csv`, in seconds from the start of the recording (the `seconds_elapsed` column in Sensor Logger), leaving about half a second of margin at every change. Time spent picking up or putting down the phone can be left out; unlabelled stretches are decoded but not scored.

```
recording,start_s,end_s,activity
test_01_member_a,0.5,9.5,still
test_01_member_a,12.0,19.5,standing
test_01_member_a,20.0,31.5,walking
```

**The videos the report needs.** Separately from the above, each activity needs a video of at least 5 seconds showing your whole body and your face, in good light, holding the phone and recording. Four videos in total, linked in the report (an unlisted YouTube link or a Google Drive link with view access). The brief says a missing video means zero for the assignment.

## 2. Running the notebook

```
pip install -r requirements.txt
jupyter notebook HMM_activity_recognition.ipynb      # then Run All
```

The first cells print a checklist (number of files, seconds per activity, test files present, sampling gaps, gravity handling). Fix anything marked `CHECK` before going further. Recordings that could not be read are listed with the reason.

Settings at the top of the notebook: `TARGET_HZ` (common sampling rate, 50), `WINDOW_S` and `STEP_S` (2 s windows, 1 s hop), `COV_TYPE` (`"diag"` or `"full"` Gaussian emissions), `BW_TOL` (Baum–Welch stopping threshold).

## 3. Where each report section gets its material

| Report section | From the notebook |
|---|---|
| Data collection and pre-processing | `results/sampling_rates_by_member.csv`, `results/seconds_per_activity.csv`, `figures/data_seconds_per_activity.png`, `figures/raw_signals_by_activity.png`, the window-size and resampling explanations in sections 1 and 2 |
| Features | the feature table in section 2, `figures/feature_distributions.png` |
| HMM setup and implementation | sections 3 and 4, `figures/baum_welch_convergence.png`, `figures/transition_matrix.png`, `figures/emission_parameters.png` |
| Results | `results/evaluation_table.md`, `figures/confusion_matrix_test.png`, `figures/decoded_*.png`, `results/decoder_comparison.csv`, `results/test_accuracy_per_file.csv` |
| Discussion | the statements printed in section 6 (also in `results/summary.json`) |
