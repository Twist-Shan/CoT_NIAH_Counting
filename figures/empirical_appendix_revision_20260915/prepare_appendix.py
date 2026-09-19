"""Generate a shorter appendix, retaining complete observations and fit scores."""
from pathlib import Path
import json
import re

import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
BASE=ROOT/'figures/empirical_section3_refit'
PAPER=ROOT/'runs/paper_figures'
BEFORE=PAPER/'output/empirical_appendix_revision_20260915/before/runs/paper_figures'
ORDER=[('Qwen3-4B','Qwen3-4B'),('Qwen3-8B','Qwen3-8B'),('Qwen3-14B','Qwen3-14B'),
       ('Qwen3-32B','Qwen3-32B'),('Gemma4-E4B','Gemma-4-E4B'),('Gemma4-12B','Gemma-4-12B'),
       ('Gemma4-26B-A4B','Gemma-4-26B-A4B'),('Gemma4-31B','Gemma-4-31B'),
       ('Nemotron-Nano-v2-9B','Nemotron Nano v2-9B'),('Nemotron-3-Nano-4B','Nemotron Nano 3-4B'),
       ('GLM-4/Z1-9B',r'GLM-4/Z1-9B$^\dagger$'),
       ('Ministral-3-8B pair',r'Ministral-3-8B pair$^\dagger$')]
FORMS=['phi_linear','phi_log','hazard_linear','hazard_log','reciprocal']


def main():
    scores=pd.read_csv(BASE/'fit_scores.csv')
    means=pd.read_csv(BASE/'original_group_accuracy.csv').set_index('model')
    current=pd.read_csv(ROOT/'figures/empirical_law_1x4/long_context_observations.csv')
    lengths=pd.read_csv(OUT/'length_scores.csv')
    model_rows=[]
    numerical=[]

    def primary_row(scope,model,title,mean):
        row=[]
        for mode,form in [('direct','gaussian_linear'),('native_thinking','interaction_free_length')]:
            r=scores.loc[scores.scope.eq(scope)&scores.model.eq(model)&scores['mode'].eq(mode)
                         &scores.form.eq(form)&scores.validation.eq('leave_count_out')]
            assert len(r)==1
            r=r.iloc[0]
            row += [f'{r.r2:.3f}',f'{r.rmse_pp:.1f}']
        model_rows.append(title+' & '+f'{100*mean["direct"]:.1f} & {100*mean["native_thinking"]:.1f} & '
                          +' & '.join(row)+r'\\')
        numerical.append({'table':'primary','scope':scope,'model':model,'values':row})

    for model,title in ORDER:
        primary_row('original_short',model,title,means.loc[model])
    model_rows.append(r'\midrule\multicolumn{7}{l}{Current 1k--100k grid}\\')
    for model,title in [('Qwen3-32B','Qwen3-32B (rerun)'),('Gemma4-31B','Gemma-4-31B')]:
        mean=current.loc[current.model.eq(model)].groupby('mode')['observed'].mean()
        primary_row('current_full',model,title,mean)

    length_rows=[]
    def length_row(scope,model,title,validation='leave_length_out'):
        values=[]
        for form in FORMS:
            r=lengths.loc[lengths.scope.eq(scope)&lengths.model.eq(model)&lengths.form.eq(form)
                          &lengths.validation.eq(validation)]
            assert len(r)==1
            r=r.iloc[0]
            values.append(f'{r.r2:.2f}/{r.rmse_pp:.1f}')
        length_rows.append(title+' & '+' & '.join(values)+r'\\')
        numerical.append({'table':'length','scope':scope,'model':model,'values':values})
    length_row('median_short','Median across 12 groups','Across-model median')
    for model,title in ORDER:
        length_row('original_short',model,title)
    length_rows.append(r'\midrule\multicolumn{6}{l}{Current 1k--100k grid: leave one length out}\\')
    for model,title in [('Qwen3-32B','Qwen3-32B (rerun)'),('Gemma4-31B','Gemma-4-31B')]:
        length_row('current_full',model,title)
    length_rows.append(r'\midrule\multicolumn{6}{l}{Frozen 1k--20k fit evaluated at 25k--100k}\\')
    for model,title in [('Qwen3-32B','Qwen3-32B (rerun)'),('Gemma4-31B','Gemma-4-31B')]:
        length_row('short_to_long',model,title,'frozen_extrapolation')

    old=(BEFORE/'appendix/behavioral-comparison.tex').read_text(encoding='utf-8')
    prefix=old.split(r'\paragraph{\gpt{Evaluation and scoring.}}')[0]
    prefix=prefix.replace(r'\subsection{Evaluation grid and model-level results}',
                          r'\subsection{\gpt{Evaluation setup}}')
    appendix=prefix+r'''\paragraph{\gpt{Grids and scoring.}}
\gpt{The original benchmark has 12 comparison groups, two modes, 14 counts
$N\in\{1,\ldots,10,12,15,18,20\}$, and eight lengths $L\in\{1,2,3,5,8,10,15,20\}$k.
Each condition contains 30 paired seeds, giving 80,640 requests.
Parsed exact accuracy requires the final integer matching \texttt{Total:} to equal $N$; unparseable responses receive zero and all requests remain in the denominator.\footnote{\gpt{The archived short-context metric includes one truncated but parsed-correct GLM Thinking response. Treating it as an error changes that group's mean by $-0.03$ percentage points.}}
The length extension adds 25, 30, 40, 50, 60, 70, 80, 90, and 100k.
Qwen3-32B was rerun over all 17 lengths with YaRN disabled (14,280 requests); Gemma-4-31B retains its original and extension batches.
These length analyses count truncations as errors, with output limits of 64 tokens for Non-thinking and 4,096 for Thinking.
Qwen's manifests verify default RoPE and a 131,072-position cache; Gemma's registration specifies no RoPE override, but its runtime manifests were unavailable.
Passage lengths use the common Qwen3-8B tokenizer.}

\paragraph{\gpt{Main-figure summaries.}}
\gpt{Panels A,B report the median and IQR across the 12 group-level accuracies at each $(N,L)$.
Their display lines use Gaussian-kernel averages in $\log_2N$, with bandwidth 0.30; the measured medians and IQRs remain unchanged.
Panels C,D join model-level means for six counts at 25k--100k.
All analyses below use unsmoothed observations and retain every count, including $N=1$.}

\subsection{\gpt{Count dependence: models, fits, and complete group results}}
\label{app:empirical-regression}
\label{app:empirical-cross-model-fits}

\paragraph{\gpt{Two approximations.}}
\gpt{For Non-thinking, take a scalar readout $Z=N\Delta+s(L)NG$, with $G\sim\mathcal N(0,1)$.
Nearest-center decoding is correct when $|Z-N\Delta|<\Delta/2$, yielding
\begin{equation}
p_{\mathrm{NT}}(N,L)=2\Phi(\kappa_L/N)-1,
\qquad \kappa_L=\Delta/[2s(L)]>0.
\label{eq:empirical-gaussian}
\end{equation}
This is the scalar-variability approximation discussed in numerical-cognition models~\citep{dehaene2003weber}.
Only the noise-to-spacing ratio is identified; attention averaging alone does not imply $\sigma\propto N$.
For Thinking, assume $N$ independent required steps with success probability $1-\varphi_L$, an independent baseline failure $\alpha$, and no cancellation of step errors. Then
\begin{equation}
p_{\mathrm{T}}(N,L)=(1-\alpha)(1-\varphi_L)^N
=\exp[-h_0-N\lambda_L],
\label{eq:empirical-step-product}
\end{equation}
where $h_0=-\ln(1-\alpha)$ and $\lambda_L=-\ln(1-\varphi_L)$ are nonnegative.
Taylor's theorem gives $0\leq\alpha+(1-\alpha)N\varphi_L-(1-p_{\mathrm{T}})\leq(1-\alpha)\binom N2\varphi_L^2$.
Thus the main-text linear error approximation requires $N\varphi_L\ll1$; all fitted predictions use the bounded product.}

\paragraph{\gpt{Estimation and validation.}}
\gpt{Nonlinear least squares fits each group separately and, separately, the descriptive cross-model medians.
For equal-size condition means, squared error is equivalent to request-level Brier loss up to a constant.
At $J$ lengths, Non-thinking has $J$ free parameters and Thinking $J+1$.
Validation holds out every condition at one count and refits (14 folds), testing new counts at measured lengths.
For observed accuracies $a_c$ and out-of-fold predictions $\widehat p_c$, $R^2=1-\sum_c(a_c-\widehat p_c)^2/\sum_c(a_c-\overline a)^2$ and $\mathrm{RMSE}=100\sqrt{\operatorname{mean}_c(a_c-\widehat p_c)^2}$ in percentage points.
The analysis is exploratory; reused seeds preclude treating conditions as independent samples for confidence intervals.}

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/empirical_appendix/empirical_median_fits.pdf}
\caption{\gpt{\textbf{Count-model fits to unsmoothed cross-model medians.} All 14 counts and eight lengths are shown. Curves use Eqs.~\eqref{eq:empirical-gaussian} and~\eqref{eq:empirical-step-product}.}}
\label{fig:empirical-median-fits}
\end{figure}

\gpt{Median training/held-count $R^2$ is $0.858/0.828$ for Non-thinking and $0.942/0.924$ for Thinking, with held-count RMSE 15.2 and 3.6 points.
The Non-thinking transition shows systematic departures.
Thinking's overall mean is higher in every group (Table~\ref{tab:empirical-model-results}); Fig.~\ref{fig:empirical-all-models} gives all group-level observations and fits.}

\begin{table}[H]
\centering
\small
\renewcommand{\arraystretch}{0.90}
\setlength{\tabcolsep}{3.5pt}
\gpt{\begin{tabular}{lrrrrrr}
\toprule
& \multicolumn{2}{c}{Mean accuracy (\%)} & \multicolumn{2}{c}{Non-thinking} & \multicolumn{2}{c}{Thinking}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Comparison group & NT & T & CV $R^2$ & RMSE & CV $R^2$ & RMSE\\
\midrule
@MODEL_ROWS@
\bottomrule
\end{tabular}}
\caption{\gpt{\textbf{Observed means and held-count fits.} The original and current grids contain 3,360 and 7,140 requests per mode per group, respectively. RMSE: percentage points. $^\dagger$Separate checkpoints.}}
\label{tab:empirical-model-results}
\end{table}

\begin{figure}[p]
\centering
\includegraphics[width=\linewidth]{figures/empirical_appendix/empirical_accuracy_12models.pdf}
\caption{\gpt{\textbf{Complete 12-group comparison at 1k--20k.} Every panel includes 14 counts, eight lengths, and both modes. Points are 30-request accuracies; solid Gaussian curves and circles denote Non-thinking, dashed step-success curves and triangles Thinking. Colors identify length. Fits are separate for each group. $^\dagger$Separate checkpoints.}}
\label{fig:empirical-all-models}
\end{figure}
\FloatBarrier

\subsection{\gpt{Complete long-context observations}}
\label{app:empirical-long-context}
\label{app:empirical-long-observations}

\gpt{Figures~\ref{fig:empirical-qwen-all-counts} and~\ref{fig:empirical-gemma-all-counts} expand main panels C,D to all 14 counts, 17 lengths, and both modes (476 conditions/model).
Shading gives pointwise 95\% Wilson intervals for 30 binary outcomes; these are not simultaneous bands or paired tests across conditions.}

\gpt{Local reversals remain visible.
At $N=1,L=100$k, Qwen's Non-thinking and Thinking accuracies are $22/30$ and $14/30$; 12 Thinking outputs exhaust the 4,096-token limit and four give incorrect totals.
Gemma Non-thinking at $N=20$ is correct on $0/30$ requests at 5k, $21/30$ at 8k, and $0/30$ at 15k.
These results delimit the overall trends: output budgets contribute to failure, and the cause of Gemma's local increase is not identified.}
\FloatBarrier

\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{figures/empirical_appendix/empirical_qwen_all_counts.pdf}
\caption{\gpt{\textbf{Qwen3-32B: all counts and lengths, YaRN disabled.} Each panel fixes $N$ and shows all 17 lengths from 1k to 100k. Rose circles/solid lines are Non-thinking; blue triangles/dashed lines are Thinking. Points are 30-seed means, shading shows pointwise 95\% Wilson intervals, and lines connect measurements without smoothing. Truncations count as errors.}}
\label{fig:empirical-qwen-all-counts}
\label{fig:empirical-full-range}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{figures/empirical_appendix/empirical_gemma_all_counts.pdf}
\caption{\gpt{\textbf{Gemma-4-31B: all counts and lengths.} The display matches Fig.~\ref{fig:empirical-qwen-all-counts}. Gray strips separate the original 1k--20k and extension 25k--100k batches; lines and intervals do not cross that gap. All 14 counts and both modes are retained, including local accuracy reversals.}}
\label{fig:empirical-gemma-all-counts}
\end{figure}
\FloatBarrier

\subsection{\gpt{Which length functions describe Thinking?}}
\label{app:empirical-length-tests}

\paragraph{\gpt{Literature and candidate functions.}}
\gpt{Length-sensitive retrieval and reasoning are documented by RULER and FLenQA~\citep{hsieh2024ruler,levy2024sametask}; these evaluations motivate testing length dependence without specifying a universal error law.
\citet{chen2026critical} derive a logarithmic scaling of attention logits in a simplified model, which concerns a different quantity from $\varphi(L)$.
We therefore compare five explicit hypotheses for the effective per-step success $q(L)=1-\varphi(L)$, keeping $p_{\mathrm{T}}=(1-\alpha)q(L)^N$.
Let $x=(L/1000-1)/99$ and $z=\ln(L/1000)/\ln100$:
\begin{equation}
\begin{aligned}
q_{\mathrm{lin}\text{-}\varphi}&=q_0(1-rx), &
q_{\log\text{-}\varphi}&=q_0(1-rz),\\
q_{\mathrm{lin}\text{-}\lambda}&=q_0e^{-bx}, &
q_{\log\text{-}\lambda}&=q_0e^{-bz}, &
q_{\mathrm{recip}}&=\frac{q_0}{1+bx}.
\end{aligned}
\label{eq:empirical-length-candidates}
\end{equation}
Here $0<q_0\leq1$, $0\leq r\leq1$, and $b\geq0$; each candidate has three parameters including $\alpha$.
The first pair makes $\varphi$ linear in $L$ or $\log L$; the second does so for $\lambda=-\ln q$.
For the reciprocal candidate, a target competing with $D$ equal-score distractors at fixed logit margin $m$ has attention mass $1/(1+De^{-m})$.
Using this shape as a success-probability proxy is an additional assumption of our model.}

\paragraph{\gpt{Comparison and scope.}}
\gpt{All candidates use the same unsmoothed accuracies and least-squares loss.
The fixed 1k--100k rescaling and parameter constraints ensure valid probabilities throughout the tested domain, including frozen short-to-long prediction; they use no held-out outcomes.
We first leave out each whole length and refit, retaining every count.
We separately train on 1k--20k and freeze all parameters before testing 25k--100k.
Figure~\ref{fig:empirical-length-candidates} shows the implied length profiles, and Table~\ref{tab:empirical-length-validation} reports every candidate and comparison group.}

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/empirical_appendix/empirical_length_candidates.pdf}
\caption{\gpt{\textbf{Candidate length dependence of effective step failure.} Dots are the free-by-length estimates from Eq.~\eqref{eq:empirical-step-product}; curves are the five joint fits in Eq.~\eqref{eq:empirical-length-candidates}. All fits use accuracy observations, not the dots. Panel A uses 1k--20k medians; B,C use each model's complete 1k--100k data. Vertical scales differ.}}
\label{fig:empirical-length-candidates}
\end{figure}

\gpt{Linear $\varphi(L)$ has lower held-length RMSE than logarithmic $\varphi(L)$ in 11 of the 12 short-context groups; Nemotron Nano 3-4B is the exception.
The median curves give 4.9 versus 9.0 points; Qwen and Gemma over 1k--100k give 15.5 versus 27.4 and 7.0 versus 13.8 points.
Linear-$\lambda$ and reciprocal-success fits have similar errors, so these results support a coarse length trend without identifying a unique functional law.
Frozen short-to-long errors remain much larger: even linear $\varphi$ gives 44.0 points for Qwen and 18.4 for Gemma.
The fitted $\varphi$ describes answer failures under the specified output budget and does not directly measure retrieval errors.}

\begin{table}[H]
\centering
\small
\setlength{\tabcolsep}{3pt}
\gpt{\begin{tabular}{lrrrrr}
\toprule
Thinking data & Linear $\varphi$ & Log $\varphi$ & Linear $\lambda$ & Log $\lambda$ & Reciprocal $q$\\
\midrule
@LENGTH_ROWS@
\bottomrule
\end{tabular}}
\caption{\gpt{\textbf{Length-model validation: $R^2$/RMSE (percentage points).} The first 13 rows leave out each length in the original 1k--20k grid; the next two do so over 1k--100k. The final two are frozen short-to-long predictions. All five candidates have three parameters and retain all 14 counts. Higher $R^2$ and lower RMSE indicate better predictions. $^\dagger$Separate checkpoints.}}
\label{tab:empirical-length-validation}
\label{tab:empirical-length-extrapolation}
\end{table}
\FloatBarrier
'''
    appendix=appendix.replace('@MODEL_ROWS@','\n'.join(model_rows)).replace('@LENGTH_ROWS@','\n'.join(length_rows))
    assert '@MODEL_ROWS@' not in appendix and '@LENGTH_ROWS@' not in appendix
    (OUT/'appendix_revised.tex').write_text(appendix,encoding='utf-8')
    (OUT/'manuscript_values.json').write_text(json.dumps(numerical,indent=2),encoding='utf-8')
    def prose_words(text):
        text='\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('%'))
        text=re.sub(r'\\begin\{(equation|tabular)\}.*?\\end\{\1\}',' ',text,flags=re.S)
        text=re.sub(r'\\(?:ref|eqref|label|citep|citet|includegraphics)\*?(?:\[[^\]]*\])?\{[^}]*\}',' ',text)
        text=re.sub(r'\\[a-zA-Z]+',' ',text)
        return len(re.findall(r'[A-Za-z]+(?:[-\u2013][A-Za-z]+)*',text))
    summary={'old_prose_word_proxy':prose_words(old),'new_prose_word_proxy':prose_words(appendix),
             'old_figures':old.count(r'\begin{figure}'),'new_figures':appendix.count(r'\begin{figure}'),
             'old_tables':old.count(r'\begin{table}'),'new_tables':appendix.count(r'\begin{table}')}
    (OUT/'revision_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
