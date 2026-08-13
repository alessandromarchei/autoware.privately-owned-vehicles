#include <IpIpoptApplication.hpp>
#include <IpSolveStatistics.hpp>
#include <IpTNLP.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

namespace {

// A tiny nonlinear lateral MPC-like problem.
// State: cross-track error cte[t], heading error epsi[t].
// Control: steering delta[t].
// Dynamics use sin() and tan(), so this is a genuine nonlinear program.
class LateralMpcNlp final : public Ipopt::TNLP {
 public:
  static constexpr Ipopt::Index kN = 6;
  static constexpr Ipopt::Index kCte = 0;
  static constexpr Ipopt::Index kEpsi = kCte + kN;
  static constexpr Ipopt::Index kDelta = kEpsi + kN;
  static constexpr Ipopt::Index kVariables = 3 * kN - 1;
  static constexpr Ipopt::Index kConstraints = 2 * kN;

  bool get_nlp_info(Ipopt::Index& n, Ipopt::Index& m,
                    Ipopt::Index& nnz_jac_g, Ipopt::Index& nnz_h_lag,
                    IndexStyleEnum& index_style) override {
    n = kVariables;
    m = kConstraints;
    // Initial cte/epsi plus 4 entries for each pair of dynamics constraints.
    nnz_jac_g = 2 + 8 * (kN - 1);
    nnz_h_lag = 0;  // Ipopt limited-memory Hessian is selected below.
    index_style = TNLP::C_STYLE;
    return true;
  }

  bool get_bounds_info(Ipopt::Index n, Ipopt::Number* x_l,
                       Ipopt::Number* x_u, Ipopt::Index m,
                       Ipopt::Number* g_l, Ipopt::Number* g_u) override {
    if (n != kVariables || m != kConstraints) return false;

    for (Ipopt::Index i = 0; i < n; ++i) {
      x_l[i] = -1.0e19;
      x_u[i] = 1.0e19;
    }
    constexpr double max_steer = 0.45;
    for (Ipopt::Index t = 0; t < kN - 1; ++t) {
      x_l[kDelta + t] = -max_steer;
      x_u[kDelta + t] = max_steer;
    }
    for (Ipopt::Index i = 0; i < m; ++i) {
      g_l[i] = 0.0;
      g_u[i] = 0.0;
    }
    g_l[0] = kInitialCte;
    g_u[0] = kInitialCte;
    return true;
  }

  bool get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number* x,
                          bool init_z, Ipopt::Number*, Ipopt::Number*,
                          Ipopt::Index, bool init_lambda,
                          Ipopt::Number*) override {
    if (n != kVariables || !init_x || init_z || init_lambda) return false;
    std::fill(x, x + n, 0.0);
    for (Ipopt::Index t = 0; t < kN; ++t) x[kCte + t] = kInitialCte;
    return true;
  }

  bool eval_f(Ipopt::Index n, const Ipopt::Number* x, bool,
              Ipopt::Number& obj_value) override {
    if (n != kVariables) return false;
    obj_value = 0.0;
    for (Ipopt::Index t = 0; t < kN; ++t) {
      obj_value += 2000.0 * x[kCte + t] * x[kCte + t];
      obj_value += 1000.0 * x[kEpsi + t] * x[kEpsi + t];
    }
    for (Ipopt::Index t = 0; t < kN - 1; ++t)
      obj_value += 100.0 * x[kDelta + t] * x[kDelta + t];
    for (Ipopt::Index t = 0; t < kN - 2; ++t) {
      const double difference = x[kDelta + t + 1] - x[kDelta + t];
      obj_value += 600.0 * difference * difference;
    }
    return true;
  }

  bool eval_grad_f(Ipopt::Index n, const Ipopt::Number* x, bool,
                   Ipopt::Number* grad_f) override {
    if (n != kVariables) return false;
    std::fill(grad_f, grad_f + n, 0.0);
    for (Ipopt::Index t = 0; t < kN; ++t) {
      grad_f[kCte + t] = 4000.0 * x[kCte + t];
      grad_f[kEpsi + t] = 2000.0 * x[kEpsi + t];
    }
    for (Ipopt::Index t = 0; t < kN - 1; ++t)
      grad_f[kDelta + t] += 200.0 * x[kDelta + t];
    for (Ipopt::Index t = 0; t < kN - 2; ++t) {
      const double difference = x[kDelta + t + 1] - x[kDelta + t];
      grad_f[kDelta + t] -= 1200.0 * difference;
      grad_f[kDelta + t + 1] += 1200.0 * difference;
    }
    return true;
  }

  bool eval_g(Ipopt::Index n, const Ipopt::Number* x, bool,
              Ipopt::Index m, Ipopt::Number* g) override {
    if (n != kVariables || m != kConstraints) return false;
    g[0] = x[kCte];
    g[1] = x[kEpsi];
    for (Ipopt::Index t = 0; t < kN - 1; ++t) {
      const Ipopt::Index row = 2 + 2 * t;
      g[row] = x[kCte + t + 1] - x[kCte + t]
             - kVelocity * std::sin(x[kEpsi + t]) * kDt;
      g[row + 1] = x[kEpsi + t + 1] - x[kEpsi + t]
                 + (kVelocity / kLf) * std::tan(x[kDelta + t]) * kDt;
    }
    return true;
  }

  bool eval_jac_g(Ipopt::Index n, const Ipopt::Number* x, bool,
                  Ipopt::Index m, Ipopt::Index nele_jac,
                  Ipopt::Index* i_row, Ipopt::Index* j_col,
                  Ipopt::Number* values) override {
    if (n != kVariables || m != kConstraints ||
        nele_jac != 2 + 8 * (kN - 1)) return false;

    Ipopt::Index p = 0;
    if (values == nullptr) {
      i_row[p] = 0; j_col[p++] = kCte;
      i_row[p] = 1; j_col[p++] = kEpsi;
      for (Ipopt::Index t = 0; t < kN - 1; ++t) {
        const Ipopt::Index row = 2 + 2 * t;
        i_row[p] = row;     j_col[p++] = kCte + t + 1;
        i_row[p] = row;     j_col[p++] = kCte + t;
        i_row[p] = row;     j_col[p++] = kEpsi + t;
        i_row[p] = row;     j_col[p++] = kDelta + t;
        i_row[p] = row + 1; j_col[p++] = kEpsi + t + 1;
        i_row[p] = row + 1; j_col[p++] = kEpsi + t;
        i_row[p] = row + 1; j_col[p++] = kDelta + t;
        // Structurally present but numerically zero: keeps four entries/row.
        i_row[p] = row + 1; j_col[p++] = kCte + t;
      }
    } else {
      values[p++] = 1.0;
      values[p++] = 1.0;
      for (Ipopt::Index t = 0; t < kN - 1; ++t) {
        values[p++] = 1.0;
        values[p++] = -1.0;
        values[p++] = -kVelocity * std::cos(x[kEpsi + t]) * kDt;
        values[p++] = 0.0;
        values[p++] = 1.0;
        values[p++] = -1.0;
        const double cosine = std::cos(x[kDelta + t]);
        values[p++] = (kVelocity / kLf) * kDt / (cosine * cosine);
        values[p++] = 0.0;
      }
    }
    return true;
  }

  bool eval_h(Ipopt::Index, const Ipopt::Number*, bool, Ipopt::Number,
              Ipopt::Index, const Ipopt::Number*, bool, Ipopt::Index,
              Ipopt::Index*, Ipopt::Index*, Ipopt::Number*) override {
    return false;  // Not called with hessian_approximation=limited-memory.
  }

  void finalize_solution(Ipopt::SolverReturn status, Ipopt::Index n,
                         const Ipopt::Number* x, const Ipopt::Number*,
                         const Ipopt::Number*, Ipopt::Index,
                         const Ipopt::Number*, const Ipopt::Number*,
                         Ipopt::Number obj_value,
                         const Ipopt::IpoptData*,
                         Ipopt::IpoptCalculatedQuantities*) override {
    solved_ = status == Ipopt::SUCCESS;
    objective_ = obj_value;
    if (n == kVariables) std::copy(x, x + n, solution_.begin());
  }

  bool solved() const { return solved_; }
  double objective() const { return objective_; }
  const std::array<double, kVariables>& solution() const { return solution_; }

 private:
  static constexpr double kDt = 0.10;
  static constexpr double kVelocity = 8.0;
  static constexpr double kLf = 2.67;
  static constexpr double kInitialCte = 1.0;

  bool solved_ = false;
  double objective_ = 0.0;
  std::array<double, kVariables> solution_{};
};

}  // namespace

int main() {
  auto problem = Ipopt::SmartPtr<LateralMpcNlp>(new LateralMpcNlp());
  Ipopt::SmartPtr<Ipopt::IpoptApplication> application =
      IpoptApplicationFactory();

  application->Options()->SetStringValue("linear_solver", "mumps");
  application->Options()->SetStringValue("hessian_approximation", "limited-memory");
  application->Options()->SetIntegerValue("print_level", 5);
  application->Options()->SetIntegerValue("max_iter", 100);
  application->Options()->SetNumericValue("tol", 1.0e-7);

  const auto init_status = application->Initialize();
  if (init_status != Ipopt::Solve_Succeeded) {
    std::cerr << "FAIL: Ipopt initialization status " << init_status << '\n';
    return 2;
  }

  const auto solve_status = application->OptimizeTNLP(problem);
  const auto& x = problem->solution();

  std::cout << std::fixed << std::setprecision(6)
            << "\n=== V4M IPOPT SMOKE TEST ===\n"
            << "status       : " << solve_status << '\n'
            << "objective    : " << problem->objective() << '\n'
            << "initial cte  : " << x[LateralMpcNlp::kCte] << '\n'
            << "final cte    : " << x[LateralMpcNlp::kCte + LateralMpcNlp::kN - 1] << '\n'
            << "first steer  : " << x[LateralMpcNlp::kDelta] << '\n';

  if (!problem->solved() || solve_status != Ipopt::Solve_Succeeded) {
    std::cerr << "FAIL: nonlinear solve did not converge\n";
    return 3;
  }
  if (!std::isfinite(problem->objective())) {
    std::cerr << "FAIL: non-finite objective\n";
    return 4;
  }

  std::cout << "PASS: Ipopt + MUMPS + BLAS solved the nonlinear MPC test\n";
  return 0;
}
