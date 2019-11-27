#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <deal.II/fe/fe_raviart_thomas.h>

using namespace dealii;

template <int dim>
class MixedPoisson
{
public:
  MixedPoisson(const unsigned int degree);
  void run();

private:
  void make_grid_and_dofs();
  void assemble_system();
  void solve();
  void compute_errors() const;
  void output_results() const;

  const unsigned int degree;

  Triangulation<dim> triangulation;
  FESystem<dim>      fe;
  DoFHandler<dim>    dof_handler;

  BlockSparsityPattern      sparsity_pattern;
  BlockSparseMatrix<double> system_matrix;

  BlockVector<double> solution;
  BlockVector<double> system_rhs;
};

template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide()
    : Function<dim>(1)
  {}

  virtual double value(const Point<dim> & p,
                       const unsigned int component = 0) const override;
};

template <int dim>
class PressureBoundaryValues : public Function<dim>
{
public:
  PressureBoundaryValues()
    : Function<dim>(1)
  {}

  virtual double value(const Point<dim> & p,
                       const unsigned int component = 0) const override;
};


template <int dim>
class ExactSolution : public Function<dim>
{
public:
  ExactSolution()
    : Function<dim>(dim + 1)
  {}

  virtual void vector_value(const Point<dim> &p,
                            Vector<double> &  value) const override;
};


template <int dim>
class PotentialBoundaryValues : public Function<dim>
{
public:
  PotentialBoundaryValues()
    : Function<dim>(1)
  {}

  virtual double value(const Point<dim> & p,
                       const unsigned int component = 0) const override
  {
      return p.square();
  };
};

template <int dim>
class ChargeDensity : public Function<dim>
{
public:
  ChargeDensity()
    : Function<dim>(1)
  {}

  virtual double value(const Point<dim> & p,
                       const unsigned int component = 0) const override
  {
      return -2.0*dim;
  };
};

template <int dim>
class ManufacturedSolution : public Function<dim>
{
public:
  ManufacturedSolution()
    : Function<dim>(dim + 1)
  {}

  virtual void vector_value(const Point<dim> &p,
                            Vector<double> &  value) const override
  {
      for (int i=0; i<dim; i++) {
          value[i] = -2*p[i];
      }
      value[dim] = p.square();
  };
};
template <int dim>
MixedPoisson<dim>::MixedPoisson(const unsigned int degree)
  : degree(degree)
  , fe(FE_RaviartThomas<dim>(degree), 1, FE_DGQ<dim>(degree), 1)
  , dof_handler(triangulation)
{}

template <int dim>
void MixedPoisson<dim>::make_grid_and_dofs()
{
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(4);

  dof_handler.distribute_dofs(fe);

  DoFRenumbering::component_wise(dof_handler);

  std::vector<types::global_dof_index> dofs_per_component(dim + 1);
  DoFTools::count_dofs_per_component(dof_handler, dofs_per_component);
  const unsigned int n_E = dofs_per_component[0],
                     n_phi = dofs_per_component[dim];

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "Total number of cells: " << triangulation.n_cells()
            << std::endl
            << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << " (" << n_E << '+' << n_phi << ')' << std::endl;

  BlockDynamicSparsityPattern dsp(2, 2);
  dsp.block(0, 0).reinit(n_E, n_E);
  dsp.block(1, 0).reinit(n_phi, n_E);
  dsp.block(0, 1).reinit(n_E, n_phi);
  dsp.block(1, 1).reinit(n_phi, n_phi);
  dsp.collect_sizes();
  DoFTools::make_sparsity_pattern(dof_handler, dsp);

  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);

  solution.reinit(2);
  solution.block(0).reinit(n_E);
  solution.block(1).reinit(n_phi);
  solution.collect_sizes();

  system_rhs.reinit(2);
  system_rhs.block(0).reinit(n_E);
  system_rhs.block(1).reinit(n_phi);
  system_rhs.collect_sizes();
}


template <int dim>
void MixedPoisson<dim>::assemble_system()
{
  QGauss<dim>     quadrature_formula(degree + 2);
  QGauss<dim - 1> face_quadrature_formula(degree + 2);

  FEValues<dim>     fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
  FEFaceValues<dim> fe_face_values(fe,
                                   face_quadrature_formula,
                                   update_values | update_normal_vectors |
                                     update_quadrature_points |
                                     update_JxW_values);

  const unsigned int dofs_per_cell   = fe.dofs_per_cell;
  const unsigned int n_q_points      = quadrature_formula.size();
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     local_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const ChargeDensity<dim>          right_hand_side;
  const PotentialBoundaryValues<dim> potential_boundary_values;

  std::vector<double>         rhs_values(n_q_points);
  std::vector<double>         boundary_values(n_face_q_points);

  const FEValuesExtractors::Vector E(0);
  const FEValuesExtractors::Scalar potential(dim);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      local_matrix = 0;
      local_rhs    = 0; right_hand_side.value_list(fe_values.get_quadrature_points(),
                                 rhs_values);

      for (unsigned int q = 0; q < n_q_points; ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            const Tensor<1, dim> phi_i_E = fe_values[E].value(i, q);
            const double div_phi_i_E = fe_values[E].divergence(i, q);
            const double phi_i_pot     = fe_values[potential].value(i, q);

            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                const Tensor<1, dim> phi_j_E =
                  fe_values[E].value(j, q);
                const double div_phi_j_E =
                  fe_values[E].divergence(j, q);
                const double phi_j_pot = fe_values[potential].value(j, q);

                local_matrix(i, j) +=
                  (phi_i_E * phi_j_E //
                   - phi_i_pot * div_phi_j_E                 //
                   - div_phi_i_E * phi_j_pot)                //
                  * fe_values.JxW(q);
              }

            local_rhs(i) += -phi_i_pot * rhs_values[q] * fe_values.JxW(q);
          }

      for (unsigned int face_n = 0;
           face_n < GeometryInfo<dim>::faces_per_cell;
           ++face_n)
        if (cell->at_boundary(face_n))
          {
            fe_face_values.reinit(cell, face_n);

            potential_boundary_values.value_list(
              fe_face_values.get_quadrature_points(), boundary_values);

            for (unsigned int q = 0; q < n_face_q_points; ++q)
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                local_rhs(i) += -(fe_face_values[E].value(i, q) * //
                                  fe_face_values.normal_vector(q) *        //
                                  boundary_values[q] *                     //
                                  fe_face_values.JxW(q));
          }

      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          system_matrix.add(local_dof_indices[i],
                            local_dof_indices[j],
                            local_matrix(i, j));
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        system_rhs(local_dof_indices[i]) += local_rhs(i);
    }
}

template <int dim>
void MixedPoisson<dim>::solve()
{
  // As a first step we declare references to all block components of the
  // matrix, the right hand side and the solution vector that we will
  // need.
  const auto &M = system_matrix.block(0, 0);
  const auto &B = system_matrix.block(0, 1);

  const auto &F = system_rhs.block(0);
  const auto &G = system_rhs.block(1);

  auto &U = solution.block(0);
  auto &P = solution.block(1);

  const auto op_M = linear_operator(M);
  const auto op_B = linear_operator(B);

  ReductionControl     reduction_control_M(2000, 1.0e-18, 1.0e-10);
  SolverCG<>           solver_M(reduction_control_M);
  PreconditionJacobi<> preconditioner_M;

  preconditioner_M.initialize(M);

  const auto op_M_inv = inverse_operator(op_M, solver_M, preconditioner_M);

  const auto op_S = transpose_operator(op_B) * op_M_inv * op_B;
  const auto op_aS =
    transpose_operator(op_B) * linear_operator(preconditioner_M) * op_B;

  IterationNumberControl iteration_number_control_aS(30, 1.e-18);
  SolverCG<>             solver_aS(iteration_number_control_aS);

  const auto preconditioner_S =
    inverse_operator(op_aS, solver_aS, PreconditionIdentity());

  const auto schur_rhs = transpose_operator(op_B) * op_M_inv * F - G;

  SolverControl solver_control_S(2000, 1.e-12);
  SolverCG<>    solver_S(solver_control_S);

  const auto op_S_inv = inverse_operator(op_S, solver_S, preconditioner_S);

  P = op_S_inv * schur_rhs;

  std::cout << solver_control_S.last_step()
            << " CG Schur complement iterations to obtain convergence."
            << std::endl;

  U = op_M_inv * (F - op_B * P);
}


template <int dim>
void MixedPoisson<dim>::compute_errors() const
{
  const ComponentSelectFunction<dim> potential_mask(dim, dim + 1);
  const ComponentSelectFunction<dim> E_mask(std::make_pair(0, dim),
                                                   dim + 1);

  ManufacturedSolution<dim> manufactured_solution;
  Vector<double>     cellwise_errors(triangulation.n_active_cells());

  QTrapez<1>     q_trapez;
  QIterated<dim> quadrature(q_trapez, degree + 2);

  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    manufactured_solution,
                                    cellwise_errors,
                                    quadrature,
                                    VectorTools::L2_norm,
                                    &potential_mask);
  const double pot_l2_error =
    VectorTools::compute_global_error(triangulation,
                                      cellwise_errors,
                                      VectorTools::L2_norm);

  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    manufactured_solution,
                                    cellwise_errors,
                                    quadrature,
                                    VectorTools::L2_norm,
                                    &E_mask);
  const double E_l2_error =
    VectorTools::compute_global_error(triangulation,
                                      cellwise_errors,
                                      VectorTools::L2_norm);

  Assert(E_l2_error < 1e-9, ExcInternalError());
  std::cout << "Errors: ||e_pot||_L2 = " << pot_l2_error
            << ",   ||e_E||_L2 = " << E_l2_error << std::endl;
}


template <int dim>
void MixedPoisson<dim>::output_results() const
{
  std::vector<std::string> solution_names(dim, "E");
  solution_names.emplace_back("pot");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation(dim,
                   DataComponentInterpretation::component_is_part_of_vector);
  interpretation.push_back(DataComponentInterpretation::component_is_scalar);

  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler,
                           solution,
                           solution_names,
                           interpretation);

  data_out.build_patches(degree + 1);

  std::stringstream sfilename;
  sfilename << "solution" << dim << "d.vtu";
  std::ofstream output(sfilename.str());
  data_out.write_vtu(output);
}

template <int dim>
void MixedPoisson<dim>::run()
{
  
    TimerOutput timer (std::cout, TimerOutput::summary,
                   TimerOutput::wall_times);
  timer.enter_subsection ("make_grid_and_dofs");
  make_grid_and_dofs();
  timer.leave_subsection();
  timer.enter_subsection ("assemble_system");
  assemble_system();
  timer.leave_subsection();
  timer.enter_subsection ("solve");
  solve();
  timer.leave_subsection();
  timer.enter_subsection ("compute_errors");
  compute_errors();
  timer.leave_subsection();
  timer.enter_subsection ("output_results");
  output_results();
  timer.leave_subsection();
}

int main()
{
  try
    {
      using namespace dealii;

      const unsigned int     fe_degree = 0;

      // there is no 1d RaviartThomas element
      std::cout << "2d" << std::endl;
      MixedPoisson<2> prob2(fe_degree);
      prob2.run();

      std::cout << "3d" << std::endl;
      MixedPoisson<3> prob3(fe_degree);
      prob3.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
