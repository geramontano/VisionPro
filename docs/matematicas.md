# Mathematical formulation

This document describes the mathematical structure of VisionPro. The system combines facial emotion recognition, gaze estimation, camera distance approximation and behavioral logging from a real time webcam stream.

## 1. Video stream and frame representation

Let the webcam stream be represented as a sequence of RGB frames:

$$
\mathcal{I} = \{ I_t \}_{t=1}^{T}
$$

where each frame is an image tensor:

$$
I_t \in \mathbb{R}^{H \times W \times 3}
$$

Here, \(H\) is the image height, \(W\) is the image width and \(t\) is the discrete time index associated with the current webcam frame.

The complete runtime system can be described as a function:

$$
\mathcal{F}: I_t \mapsto (\hat{y}_t, c_t, \hat{s}_t, \hat{d}_t, z_t)
$$

where:

- \(\hat{y}_t\) is the predicted emotion.
- \(c_t\) is the live confidence score.
- \(\hat{s}_t\) is the estimated gaze position on the screen.
- \(\hat{d}_t\) is the estimated camera to face distance.
- \(z_t\) is the screen region or attention zone.

## 2. Face detection and face crop

A face detector is applied to each frame:

$$
D(I_t) = b_t
$$

where \(b_t\) is the detected face bounding box:

$$
b_t = (x_t^{min}, y_t^{min}, x_t^{max}, y_t^{max})
$$

If no face is detected, the system does not perform emotion inference for that frame.

When a face is detected, the facial region is cropped as:

$$
F_t = \operatorname{crop}(I_t, b_t)
$$

The crop is then normalized into a standard size:

$$
\tilde{F}_t = \operatorname{resize}(F_t, h_f, w_f)
$$

where \(h_f\) and \(w_f\) are the height and width used by the feature extraction pipeline.

## 3. Feature extraction

The model does not classify the raw image directly. Instead, it maps the normalized face crop into a numerical feature vector:

$$
x_t = \phi(\tilde{F}_t)
$$

The feature map \(\phi\) is composed of three main groups of features:

$$
\phi(\tilde{F}_t)
=
[
\phi_{HOG}(\tilde{F}_t),
\phi_{LBP}(\tilde{F}_t),
\phi_{geo}(\tilde{F}_t)
]
$$

Therefore:

$$
x_t \in \mathbb{R}^{n}
$$

where \(n\) is the total number of extracted features.

## 4. HOG feature representation

The Histogram of Oriented Gradients component represents local shape through image gradients.

Given a grayscale face image \(G_t\), the horizontal and vertical gradients are approximated by:

$$
G_x(i,j) = G_t(i,j+1) - G_t(i,j-1)
$$

$$
G_y(i,j) = G_t(i+1,j) - G_t(i-1,j)
$$

The gradient magnitude is:

$$
m(i,j) = \sqrt{G_x(i,j)^2 + G_y(i,j)^2}
$$

The gradient orientation is:

$$
\theta(i,j) = \arctan \left( \frac{G_y(i,j)}{G_x(i,j)} \right)
$$

The HOG descriptor builds histograms of orientations over local image cells. For each cell \(C\), the histogram bin \(k\) is computed as:

$$
h_k(C) =
\sum_{(i,j)\in C}
m(i,j)\mathbb{1}(\theta(i,j)\in B_k)
$$

where \(B_k\) is the angular interval associated with bin \(k\).

The HOG vector is then obtained by concatenating normalized cell histograms:

$$
\phi_{HOG}(\tilde{F}_t)
=
[h_1, h_2, \ldots, h_q]
$$

This captures local facial structure such as edges, wrinkles, mouth contours and eyebrow orientation.

## 5. LBP texture representation

The Local Binary Pattern component captures local texture information.

For a pixel with intensity \(g_c\) and \(P\) neighboring pixels \(g_p\), the LBP code is:

$$
LBP_{P,R}
=
\sum_{p=0}^{P-1}
s(g_p-g_c)2^p
$$

where:

$$
s(u)=
\begin{cases}
1, & u \geq 0 \\
0, & u < 0
\end{cases}
$$

The LBP descriptor is obtained by computing a histogram of LBP codes over the face image:

$$
\phi_{LBP}(\tilde{F}_t)
=
[
n_0,
n_1,
\ldots,
n_{K-1}
]
$$

where \(n_k\) is the frequency of the \(k\)-th LBP pattern.

This descriptor helps represent fine facial texture patterns that may be associated with expression changes.

## 6. Geometric facial features

The geometric feature component represents facial proportions. These features are intended to be less dependent on absolute image size.

Let a set of facial or image based reference points be represented as:

$$
P_t = \{p_1, p_2, \ldots, p_m\}
$$

where:

$$
p_i = (u_i, v_i)
$$

A generic distance between two facial points is:

$$
d(p_i,p_j)=
\sqrt{(u_i-u_j)^2+(v_i-v_j)^2}
$$

To reduce dependence on face scale, geometric measurements are normalized by a reference facial size \(r_t\), for example face width or face height:

$$
\rho_{ij}(t)=
\frac{d(p_i,p_j)}{r_t}
$$

Examples of geometric ratios include:

$$
\rho_{mouth}(t)=
\frac{h_{mouth}(t)}{w_{face}(t)}
$$

$$
\rho_{eye}(t)=
\frac{h_{eye}(t)}{w_{face}(t)}
$$

$$
\rho_{brow}(t)=
\frac{d_{brow-eye}(t)}{h_{face}(t)}
$$

The complete geometric vector is:

$$
\phi_{geo}(\tilde{F}_t)
=
[
\rho_1(t),
\rho_2(t),
\ldots,
\rho_r(t)
]
$$

These measurements help distinguish expressions such as surprise, fear, anger and happiness.

## 7. Emotion label space

The emotion classification problem is defined over six classes:

$$
\mathcal{Y}
=
\{
y_1,y_2,y_3,y_4,y_5,y_6
\}
$$

where:

$$
\mathcal{Y}
=
\{
neutral,
happy,
sad,
angry,
afraid,
surprised
\}
$$

For each frame, the goal is to estimate:

$$
p(y \mid x_t)
$$

for every:

$$
y \in \mathcal{Y}
$$

## 8. Ensemble emotion classifier

VisionPro uses a statistical ensemble classifier instead of a single model.

Let:

$$
M_1, M_2, \ldots, M_K
$$

be the individual classifiers. In this project, the ensemble includes:

$$
M_1 = SVM_{RBF}
$$

$$
M_2 = ExtraTrees
$$

$$
M_3 = RandomForest
$$

$$
M_4 = LogisticRegression
$$

$$
M_5 = MLP
$$

Each classifier produces a probability distribution over the emotion classes:

$$
p_k(y \mid x_t)=M_k(x_t)
$$

The ensemble probability is computed by soft voting:

$$
p_{ens}(y \mid x_t)
=
\sum_{k=1}^{K}
\alpha_k p_k(y \mid x_t)
$$

where \(\alpha_k\) is the weight of the \(k\)-th classifier and:

$$
\sum_{k=1}^{K}\alpha_k=1
$$

When equal weights are used:

$$
\alpha_k = \frac{1}{K}
$$

The raw predicted emotion is:

$$
\hat{y}_t^{raw}
=
\arg\max_{y\in\mathcal{Y}}
p_{ens}(y \mid x_t)
$$

## 9. Auxiliary rule based correction

In addition to the statistical model, the runtime system can use facial rules derived from geometric measurements.

Let:

$$
r_t(y)
$$

be a rule based score for emotion \(y\), computed from facial measurements such as mouth opening, eye openness and eyebrow behavior.

The model probability and rule score can be combined as:

$$
p_{mix}(y \mid x_t)
=
\lambda p_{ens}(y \mid x_t)
+
(1-\lambda) r_t(y)
$$

where:

$$
0 \leq \lambda \leq 1
$$

If \(\lambda\) is close to 1, the statistical model dominates. If \(\lambda\) is smaller, the rule based component has more influence.

The mixed probability vector is normalized as:

$$
\tilde{p}_{mix}(y \mid x_t)
=
\frac{
p_{mix}(y \mid x_t)
}{
\sum_{y'\in\mathcal{Y}}p_{mix}(y' \mid x_t)
}
$$

## 10. Temporal smoothing

Frame by frame webcam predictions can fluctuate because of lighting, head movement or small variations in the detected face crop.

Let:

$$
p_t =
[
p_t(y_1),
p_t(y_2),
\ldots,
p_t(y_6)
]
$$

be the probability vector at frame \(t\).

A temporal window of length \(m\) is used:

$$
\mathcal{W}_t =
\{
p_t,p_{t-1},\ldots,p_{t-m+1}
\}
$$

The smoothed probability vector is:

$$
\bar{p}_t
=
\sum_{i=0}^{m-1}w_i p_{t-i}
$$

where:

$$
\sum_{i=0}^{m-1}w_i=1
$$

and:

$$
w_i \geq 0
$$

The final emotion prediction is:

$$
\hat{y}_t
=
\arg\max_{y\in\mathcal{Y}}
\bar{p}_t(y)
$$

This smoothing step improves stability during live webcam execution.

## 11. Live confidence score

The live confidence shown on screen is not the same as validation accuracy.

Let:

$$
p_{(1)} = \max_{y\in\mathcal{Y}}\bar{p}_t(y)
$$

be the highest class probability, and let:

$$
p_{(2)}
$$

be the second highest class probability.

The probability margin is:

$$
\Delta_t = p_{(1)} - p_{(2)}
$$

A live confidence score can be defined as:

$$
c_t =
\operatorname{clip}
(
p_{(1)} + \beta \Delta_t,
0,
1
)
$$

where \(\beta\) is a scaling factor.

This means that high confidence requires not only a high top probability, but also separation from the second most likely emotion.

## 12. Validation accuracy

The validation accuracy is computed as:

$$
Acc =
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}
(
\hat{y}_i = y_i
)
$$

where:

- \(N\) is the number of validation samples.
- \(y_i\) is the true label.
- \(\hat{y}_i\) is the predicted label.
- \(\mathbb{1}(\cdot)\) is the indicator function.

The stored validation accuracy of the trained model was:

$$
Acc = 0.9333
$$

or:

$$
Acc = 93.33\%
$$

This metric measures performance on the validation set. It is different from the live confidence score displayed during webcam execution.

## 13. Confusion matrix

For a multiclass classifier, the confusion matrix is:

$$
C \in \mathbb{N}^{6\times 6}
$$

where each element is:

$$
C_{ij}
=
\#\{
samples\ with\ true\ class\ y_i\ predicted\ as\ y_j
\}
$$

A perfect classifier would produce a diagonal matrix:

$$
C_{ij}=0
\quad
for
\quad
i\neq j
$$

The confusion matrix helps identify which emotion classes are more frequently confused.

## 14. Precision, recall and F1 score

For each class \(y\), precision is:

$$
Precision(y)
=
\frac{TP_y}{TP_y + FP_y}
$$

Recall is:

$$
Recall(y)
=
\frac{TP_y}{TP_y + FN_y}
$$

The F1 score is the harmonic mean:

$$
F1(y)
=
2
\frac{
Precision(y)Recall(y)
}{
Precision(y)+Recall(y)
}
$$

These metrics are useful because emotion datasets may not be perfectly balanced.

## 15. Neutral calibration

In live use, the user's neutral face may not look neutral to the general model. Some users have a serious neutral expression that may resemble sadness, anger or fear.

During neutral calibration, the system records a sequence of probability vectors while the user maintains a neutral expression:

$$
\mathcal{N}
=
\{
p_1^{neutral},
p_2^{neutral},
\ldots,
p_q^{neutral}
\}
$$

The neutral reference pattern is estimated using the component wise median:

$$
p_0^{neutral}
=
\operatorname{median}
(\mathcal{N})
$$

A compatibility score between the current probability vector \(p_t\) and the user's neutral profile can be defined as:

$$
s_{neutral}(t)
=
\exp
\left(
-
\frac{
\|p_t-p_0^{neutral}\|_2^2
}{
2\sigma^2
}
\right)
$$

If the compatibility with the neutral profile is high, the system can increase the neutral probability. This makes the system more personalized for the user.

## 16. Distance estimation

Distance is estimated from the apparent size of the detected face.

Let:

$$
h_t
$$

be the detected face height in pixels at time \(t\).

During calibration, the user is placed at a known physical distance:

$$
d_0
$$

and the corresponding detected face height is:

$$
h_0
$$

Assuming a pinhole camera approximation, apparent size is inversely proportional to distance:

$$
h_t \propto \frac{1}{d_t}
$$

Therefore:

$$
d_t =
d_0
\frac{h_0}{h_t}
$$

where:

- \(d_t\) is the estimated current distance.
- \(d_0\) is the known calibration distance.
- \(h_0\) is the face height during calibration.
- \(h_t\) is the current detected face height.

This is an approximation and depends on stable face detection, head pose and camera placement.

## 17. Gaze feature vector

The gaze module estimates the region of the screen that the user is looking at.

Let the gaze feature extraction function be:

$$
g_t = \psi(I_t)
$$

where \(g_t\) may include:

- pupil location inside the eye region,
- eye bounding box proportions,
- face position,
- relative eye position,
- normalized coordinates.

Thus:

$$
g_t \in \mathbb{R}^{m}
$$

## 18. Multipoint gaze calibration

During calibration, the user looks at known screen points:

$$
s_i = (u_i,v_i)
$$

where:

- \(u_i\) is the horizontal screen coordinate.
- \(v_i\) is the vertical screen coordinate.

For every calibration point, the system records a gaze feature vector:

$$
g_i = \psi(I_i)
$$

The calibration dataset is:

$$
\mathcal{G}
=
\{
(g_i,s_i)
\}_{i=1}^{n}
$$

The goal is to learn a mapping:

$$
G(g_t)=\hat{s}_t
$$

where:

$$
\hat{s}_t=(\hat{u}_t,\hat{v}_t)
$$

is the estimated gaze point.

## 19. Polynomial gaze regression

To improve over a purely linear mapping, the gaze features can be expanded:

$$
\Phi(g_t)
=
[
1,
g_1,
g_2,
\ldots,
g_m,
g_1^2,
g_2^2,
\ldots,
g_m^2,
g_1g_2,
\ldots
]
$$

The gaze regression model is:

$$
\hat{s}_t
=
W^\top \Phi(g_t)
$$

where:

$$
W \in \mathbb{R}^{q \times 2}
$$

contains the regression coefficients for horizontal and vertical gaze coordinates.

Using ridge regression, \(W\) can be estimated as:

$$
W
=
(\Phi^\top \Phi + \lambda I)^{-1}
\Phi^\top S
$$

where:

- \(\Phi\) is the design matrix of expanded gaze features.
- \(S\) is the matrix of known screen coordinates.
- \(\lambda\) is a regularization coefficient.
- \(I\) is the identity matrix.

## 20. Gaze smoothing

As with emotion prediction, gaze estimation can fluctuate frame by frame.

Let:

$$
\hat{s}_t = (\hat{u}_t,\hat{v}_t)
$$

be the current gaze estimate.

A smoothed gaze point can be computed using an exponential moving average:

$$
\bar{s}_t
=
\gamma \hat{s}_t
+
(1-\gamma)\bar{s}_{t-1}
$$

where:

$$
0 \leq \gamma \leq 1
$$

Higher values of \(\gamma\) react faster but are noisier. Lower values are smoother but slower.

## 21. Screen zone classification

The estimated screen point is converted into relative coordinates:

$$
r_x =
\frac{\hat{u}_t}{W_s}
$$

$$
r_y =
\frac{\hat{v}_t}{H_s}
$$

where \(W_s\) and \(H_s\) are the screen width and height.

The horizontal region is:

$$
Z_x =
\begin{cases}
left, & r_x < \frac{1}{3} \\
center, & \frac{1}{3} \leq r_x < \frac{2}{3} \\
right, & r_x \geq \frac{2}{3}
\end{cases}
$$

The vertical region is:

$$
Z_y =
\begin{cases}
top, & r_y < \frac{1}{3} \\
middle, & \frac{1}{3} \leq r_y < \frac{2}{3} \\
bottom, & r_y \geq \frac{2}{3}
\end{cases}
$$

The final screen zone is:

$$
z_t = (Z_x,Z_y)
$$

Examples include:

$$
(left,top),
(center,middle),
(right,bottom)
$$

## 22. Behavioral logging

At each time step, the system can store:

$$
\ell_t =
(
\tau_t,
\hat{y}_t,
c_t,
\hat{u}_t,
\hat{v}_t,
r_x,
r_y,
z_t,
\hat{d}_t
)
$$

where:

- \(\tau_t\) is the timestamp.
- \(\hat{y}_t\) is the predicted emotion.
- \(c_t\) is the confidence score.
- \((\hat{u}_t,\hat{v}_t)\) is the gaze position.
- \((r_x,r_y)\) is the relative gaze position.
- \(z_t\) is the screen zone.
- \(\hat{d}_t\) is the estimated distance.

The full log is:

$$
\mathcal{L}
=
\{
\ell_t
\}_{t=1}^{T}
$$

## 23. Emotion distribution by screen zone

For a given screen region \(R\), the empirical probability of emotion \(y\) is:

$$
P(y \mid R)
=
\frac{
\sum_{t=1}^{T}
\mathbb{1}(z_t=R)
\mathbb{1}(\hat{y}_t=y)
}{
\sum_{t=1}^{T}
\mathbb{1}(z_t=R)
}
$$

This allows the system to estimate which emotions are most frequently associated with a screen region.

## 24. Average confidence by screen zone

For a region \(R\), the average confidence is:

$$
\bar{c}_R
=
\frac{
\sum_{t=1}^{T}
c_t
\mathbb{1}(z_t=R)
}{
\sum_{t=1}^{T}
\mathbb{1}(z_t=R)
}
$$

This gives a measure of how stable or confident the model was when the user looked at a specific zone.

## 25. Attention time by region

The amount of time spent looking at region \(R\) can be approximated as:

$$
T_R =
\sum_{t=1}^{T}
\mathbb{1}(z_t=R)\Delta t
$$

where \(\Delta t\) is the approximate time difference between frames.

The relative attention share is:

$$
A_R =
\frac{T_R}{\sum_{R'}T_{R'}}
$$

This can be used to compare visual attention across different screen regions.

## 26. Interpretation of the complete system

The complete system can be summarized as:

$$
I_t
\rightarrow
F_t
\rightarrow
x_t
\rightarrow
\bar{p}_t
\rightarrow
\hat{y}_t
$$

for emotion recognition, and:

$$
I_t
\rightarrow
g_t
\rightarrow
\hat{s}_t
\rightarrow
z_t
$$

for gaze estimation.

The synchronized output is:

$$
(
\hat{y}_t,
c_t,
\hat{s}_t,
z_t,
\hat{d}_t
)
$$

This allows VisionPro to associate facial emotion predictions with approximate visual attention.

## 27. Limitations of the mathematical model

The mathematical model is an approximation of visible behavior. It should not be interpreted as a direct measurement of psychological state.

Important limitations include:

- facial expressions are not equivalent to internal emotions,
- webcam based gaze estimation is approximate,
- distance estimation depends on face detection stability,
- lighting and head pose affect feature extraction,
- calibration quality strongly affects gaze and distance estimation.

Therefore, the output of VisionPro should be understood as a computational estimate of facial expression and visual attention, not as a clinical or psychological assessment.
