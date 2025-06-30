Credit Scoring Business Understanding
1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Capital Accord places strong emphasis on the accurate measurement and management of credit risk. It requires financial institutions to hold capital in proportion to their risk exposure, which must be calculated using transparent and defensible methods. In this context, model interpretability is not optional—it is essential. Regulatory bodies must be able to understand, validate, and audit the credit risk models. Therefore, models must be well-documented, reproducible, and explainable, ensuring stakeholders can trace how input data leads to risk assessments. This mandates the use of interpretable modeling techniques and comprehensive documentation throughout the model development lifecycle.

2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

In many real-world datasets, a direct label indicating whether a customer defaulted on a loan is not available. To address this, we must construct a proxy variable—for example, defining default as being over 90 days past due. While this allows for supervised learning, it introduces uncertainty and noise, as the proxy might not perfectly capture true default behavior.

The business risks include:

    Misclassification: Over- or underestimating risk due to poor proxy quality.

    Bias amplification: Reinforcing historical lending biases if the proxy is based on unrepresentative data.

    Regulatory risk: Using inaccurate proxies can lead to poor decision-making and regulatory non-compliance, especially if challenged in an audit.

Hence, the proxy must be carefully justified, aligned with domain knowledge, and transparently documented.

3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

| Aspect                    | Simple (e.g., Logistic Regression + WoE)     | Complex (e.g., Gradient Boosting)                                        |
| ------------------------- | -------------------------------------------- | ------------------------------------------------------------------------ |
| **Interpretability**      | High – easy to explain to stakeholders       | Low – often a black-box                                                  |
| **Regulatory compliance** | Easier – aligns with regulatory expectations | Challenging – may require additional interpretability tools (e.g., SHAP) |
| **Performance**           | May underperform on complex data             | High predictive accuracy                                                 |
| **Auditability**          | Straightforward                              | Requires additional layers of validation                                 |
| **Deployment ease**       | Simple to deploy and maintain                | Requires more infrastructure and monitoring                              |
