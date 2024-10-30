### 轮腿式运动仿真环境

#### 致谢

`Wheel-Legged-Gym` 的实现依赖于 [legged_gym](https://github.com/leggedrobotics/legged_gym) 和 [rsl_rl](https://github.com/leggedrobotics/rsl_rl) 项目的资源，这些项目由 [Robotic Systems Lab](https://rsl.ethz.ch/) 创建。

相关链接：

- [Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://arxiv.org/abs/2109.11978)
- [Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion](https://arxiv.org/abs/2202.05481)

#### 安装步骤

1. 创建一个新的 Python 虚拟环境，使用 Python 3.6、3.7 或 3.8（推荐 3.8）。
2. 从 [https://pytorch.org/get-started/](https://pytorch.org/get-started/) 安装带有 CUDA 的 PyTorch。
3. 安装 Isaac Gym
   - 从 [https://developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym) 下载并安装 Isaac Gym Preview 4。
   - 运行以下命令：  
     ```bash
     cd isaacgym/python && pip install -e .
     ```
   - 运行示例：  
     ```bash
     cd examples && python 1080_balls_of_solitude.py
     ```
   - 如遇到问题，请查阅文档 `isaacgym/docs/index.html`。
4. 安装 `wheel_legged_gym`
   - 克隆此存储库
   - 运行以下命令：  
     ```bash
     cd Wheel-Legged-Gym && pip install -e .
     ```

#### 代码结构

1. 每个环境由一个环境文件（例如 `legged_robot.py`）和一个配置文件（例如 `legged_robot_config.py`）定义。配置文件包含两个类：一个类包含所有环境参数（`LeggedRobotCfg`），另一个类包含训练参数（`LeggedRobotCfgPPO`）。
2. 环境类和配置类均使用继承。
3. 在 `cfg` 中定义的每个非零奖励权重将添加一个相应名称的函数到元素列表中，这些元素将被求和以获取总奖励。
4. 任务必须使用 `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)` 进行注册。该注册通常在 `envs/__init__.py` 中完成，也可以在此存储库外完成。

#### 使用方法

1. 训练  
   ```bash
   python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_flat
   ```
   - 如需在 CPU 上运行，添加以下参数：`--sim_device=cpu` 和 `--rl_device=cpu`（可以选择在 CPU 上模拟并在 GPU 上进行 RL）。
   - 如需无渲染运行，添加 `--headless`。
   - **重要**：为了提高性能，开始训练后按 `v` 键关闭渲染，稍后可以启用查看进度。
   - 训练的策略会保存在 `logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt` 文件夹中，`<experiment_name>` 和 `<run_name>` 在训练配置中定义。
   - 使用 TensorBoard 监控训练进程：  
     ```bash
     tensorboard --logdir=./ --port=8080
     ```
   - 以下命令行参数会覆盖配置文件中的值：
     - `--task TASK`：任务名称。
     - `--resume`：从检查点继续训练。
     - `--experiment_name EXPERIMENT_NAME`：实验名称。
     - `--run_name RUN_NAME`：运行名称。
     - `--load_run LOAD_RUN`：要加载的运行名称（当 `resume=True` 时）。若为 -1 则加载最后一次运行。
     - `--checkpoint CHECKPOINT`：保存的模型检查点编号。若为 -1 则加载最后一个检查点。
     - `--num_envs NUM_ENVS`：创建的环境数量。
     - `--seed SEED`：随机种子。
     - `--max_iterations MAX_ITERATIONS`：最大训练迭代次数。
     - `--exptid EXPTID`：实验 ID。
   
2. 运行已训练的策略  
   ```bash
   python wheel_legged_gym/scripts/play.py --task=wheel_legged_vmc_flat
   ```
   - 默认加载最后一次运行的最新模型。可以通过在训练配置中设置 `load_run` 和 `checkpoint` 来选择其他运行/模型迭代。

3. 已有的任务
   - `wheel_legged`：在不同地形上进行开链机器人端到端训练。
   - `wheel_legged_vmc`：使用 VMC 统一开链和闭链机构的运动控制，便于将策略部署到闭链机器人上。
   - `wheel_legged_vmc_flat`：在平坦地形上训练机器人（低显存需求）。

#### 添加新环境

基础环境 `legged_robot` 实现了粗糙地形的步态任务，对应的配置不指定机器人模型（URDF/MJCF）且没有奖励权重。

1. 向 `envs/` 添加一个文件夹，包含 `<your_env>_config.py` 文件，继承自现有的环境配置。
2. 若添加新机器人：
   - 将相应的模型添加到 `resources/`。
   - 在 `cfg` 中设置模型路径，定义关节名称、默认关节位置和 PD 增益。指定 `train_cfg` 和环境名称（Python 类）。
   - 在 `train_cfg` 中设置 `experiment_name` 和 `run_name`。
3. （若需要）在 `<your_env>.py` 中实现环境，继承自现有环境，覆盖所需的函数并/或添加奖励函数。
4. 在 `isaacgym_anymal/envs/__init__.py` 中注册你的环境。
5. 根据需要修改/调整 `cfg`、`cfg_train` 中的其他参数。将奖励的权重设为 0 可以移除该奖励。请勿修改其他环境的参数！

#### 故障排除

1. 若出现 `ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory` 错误，执行以下命令：  
   ```bash
   sudo apt install libpython3.8
   ```
   对于 Conda 用户，可能还需要执行：  
   ```bash
   export LD_LIBRARY_PATH=/path/to/conda/envs/your_env/lib
   ```

#### 已知问题

1. 当在 GPU 上使用三角网格地形进行模拟时，`net_contact_force_tensor` 报告的接触力不可靠。解决方法是使用力传感器，但力会通过连续体传递，导致不理想的行为。对于腿式机器人，可以在足部/末端执行器上添加传感器以获得预期结果。在使用力传感器时，确保使用 `sensor_options.enable_forward_dynamics_forces` 排除重力影响。例如：
   ```python
   sensor_pose = gymapi.Transform()
   for name in feet_names:
       sensor_options = gymapi.ForceSensorProperties()
       sensor_options.enable_forward_dynamics_forces = False  # 例如排除重力
       sensor_options.enable_constraint_solver_forces = True  # 例如包括接触力
       sensor_options.use_world_frame = True  # 在世界坐标系中报告力（更容易获得垂直分量）
       index = self.gym.find_asset_rigid_body_index(robot_asset, name)
       self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)

   sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
   self.gym.refresh_force_sensor_tensor(self.sim)
   force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
   self.sensor_forces = force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]

   self.gym.refresh_force_sensor_tensor(self.sim)
   contact = self.sensor_forces[:, :, 2] > 1.
   ```