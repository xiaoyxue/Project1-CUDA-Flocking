#define GLM_FORCE_CUDA

#include <cuda.h>
#include "kernel.h"
#include "utilityCore.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>

#include <glm/glm.hpp>

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// Part2.1 Helper buffers
std::unique_ptr<int[]> particleArrayIndices;
std::unique_ptr<int[]> particleGridIndices;
std::unique_ptr<int[]> gridCellStartIndices;
std::unique_ptr<int[]> gridCellEndIndices;
std::unique_ptr<glm::vec3[]> pos;

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  cudaMemset(dev_vel1, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("Initialize dev_vel1 failed!");
  
  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaDeviceSynchronize();

  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");
  cudaMemset(dev_particleArrayIndices, 0, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMemset dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");
  cudaMemset(dev_particleGridIndices, 0, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMemset dev_particleGridIndices failed!");

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");
  cudaMemset(dev_gridCellStartIndices, 0, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMemset dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");
  cudaMemset(dev_gridCellEndIndices, 0, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMemset dev_gridCellEndIndices failed!");

  cudaDeviceSynchronize();

  // init cpu helper buffers
  particleArrayIndices = std::make_unique<int[]>(numObjects);
  particleGridIndices = std::make_unique<int[]>(numObjects);
  gridCellStartIndices = std::make_unique<int[]>(gridCellCount);
  gridCellEndIndices = std::make_unique<int[]>(gridCellCount);
  pos = std::make_unique<glm::vec3[]>(numObjects);
  cudaMemcpy(pos.get(), dev_pos, numObjects * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
  std::fill(gridCellStartIndices.get(), gridCellStartIndices.get() + gridCellCount, -1);
  std::fill(gridCellEndIndices.get(), gridCellEndIndices.get() + gridCellCount, -1);
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

__device__ glm::vec3 rule1(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
    glm::vec3 perceived_center(0.0f, 0.0f, 0.0f);
    int neighbor_count = 0;

    for (int i = 0; i < N; i++) {
        if (i == iSelf) {
            continue;
        }
        if (glm::distance(pos[i], pos[iSelf]) < rule1Distance) {
          perceived_center += pos[i];
          neighbor_count++;
        }
    }
    if (neighbor_count > 0) {
        perceived_center /= neighbor_count;
    }

    return neighbor_count > 0 ? (perceived_center - pos[iSelf]) * rule1Scale : glm::vec3(0.0f, 0.0f, 0.0f);
}

__device__ glm::vec3 rule2(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
    glm::vec3 c(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < N; i++) {
        if (i == iSelf) {
            continue;
        }
        if (glm::distance(pos[i], pos[iSelf]) < rule2Distance) {
            c -= (pos[i] - pos[iSelf]);
        }
    }
    return c * rule2Scale;
}

__device__ glm::vec3 rule3(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
    glm::vec3 perceived_velocity(0.0f, 0.0f, 0.0f);
    int neighbor_count = 0;
    for (int i = 0; i < N; i++) {
        if (i == iSelf) {
            continue;
        }
        if (glm::distance(pos[i], pos[iSelf]) < rule3Distance) {
            perceived_velocity += vel[i];
            neighbor_count++;
        }
    }
    if (neighbor_count > 0) {
        perceived_velocity /= neighbor_count;
    }
    return neighbor_count > 0 ? perceived_velocity * rule3Scale : glm::vec3(0.0f, 0.0f, 0.0f);
}

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
  // Rule 2: boids try to stay a distance d away from each other
  // Rule 3: boids try to match the speed of surrounding boids
  glm::vec3 velocityChange(0.0f, 0.0f, 0.0f);
  velocityChange += rule1(N, iSelf, pos, vel);
  velocityChange += rule2(N, iSelf, pos, vel);
  velocityChange += rule3(N, iSelf, pos, vel);

  return velocityChange;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  // Compute a new velocity based on pos and vel1
  // Clamp the speed
  // Record the new velocity into vel2. Question: why NOT vel1?
  unsigned int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }

  glm::vec3 newVel = vel1[index];
  newVel += computeVelocityChange(N, index, pos, vel1);

  // Clamp the speed
  if (glm::length(newVel) > maxSpeed) {
    newVel = glm::normalize(newVel) * maxSpeed;
  }

  vel2[index] = newVel;
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

int cpuGridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
  
    // Compute the 1D index of the grid cell that this boid belongs to.
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < N) {
      indices[index] = index;
    }

    // Compute the 3D grid cell coordinates of this boid.
    glm::ivec3 gridCell = glm::floor((pos[index] - gridMin) * inverseCellWidth);
    int x = gridCell.x;
    int y = gridCell.y;
    int z = gridCell.z;
    int gridIndex = gridIndex3Dto1D(x, y, z, gridResolution);
    if (index < N) {
      gridIndices[index] = gridIndex;
    }
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= N) {
    return;
  }
  if (index == 0) {
    gridCellStartIndices[particleGridIndices[index]] = index;
  }
  if (index == N - 1) {
    gridCellEndIndices[particleGridIndices[index]] = index;
  }
  if (index < N - 1 && particleGridIndices[index] != particleGridIndices[index + 1]) {
    gridCellEndIndices[particleGridIndices[index]] = index;
    gridCellStartIndices[particleGridIndices[index + 1]] = index + 1;
  }
}

__device__ glm::vec3 gridRule1(int iSelf, int startIndex, int endIndex, const glm::vec3 *pos, const glm::vec3 *vel) {
  glm::vec3 perceivedCenter = glm::vec3(0.0f, 0.0f, 0.0f);
  int neighborCount = 0;

  for (int index = startIndex; index <= endIndex; index++) {
    if (index == iSelf) {
      continue;
    }
    if (glm::distance(pos[index], pos[iSelf]) < rule1Distance) { 
      perceivedCenter += pos[index];
      neighborCount++;
    }
  }

  if (neighborCount > 0) {
    perceivedCenter /= neighborCount;
  }

  return neighborCount > 0 ? (perceivedCenter - pos[iSelf]) * rule1Scale : glm::vec3(0.0f, 0.0f, 0.0f);
}

__device__ glm::vec3 gridComputeVelocityChange(int iSelf, int startIndex, int endIndex, const glm::vec3 *pos, const glm::vec3 *vel) {
  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
  // Rule 2: boids try to stay a distance d away from each other
  // Rule 3: boids try to match the speed of surrounding boids
  glm::vec3 velocityChange(0.0f, 0.0f, 0.0f);
  velocityChange += gridRule1(iSelf, startIndex, endIndex, pos, vel);

  return velocityChange;
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

  int particleIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (particleIndex >= N) {
    return;
  }
  glm::vec3 relativePos = (pos[particleIndex] - gridMin) * inverseCellWidth;
  glm::ivec3 gridCell = glm::floor(relativePos);
  glm::vec3 frac = relativePos - glm::vec3(gridCell);
  glm::ivec3 offset = glm::ivec3(
    frac.x > 0.5f ? 1 : 0,
    frac.y > 0.5f ? 1 : 0,
    frac.z > 0.5f ? 1 : 0
  );
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        glm::ivec3 neighborCell = gridCell + glm::ivec3(i, j, k) - offset;
        // Process neighborCell as needed
        int gridIndex = gridIndex3Dto1D(neighborCell.x, neighborCell.y, neighborCell.z, gridResolution);
        int startIndex = gridCellStartIndices[gridIndex];
        int endIndex = gridCellEndIndices[gridIndex];

      }
    }
  }
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  // TODO-1.2 ping-pong the velocity buffers
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  kernUpdateVelocityBruteForce<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, dev_vel1, dev_vel2);
  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);

  // TODO-1.2 - ping-pong the velocity buffers
  std::swap(dev_vel1, dev_vel2);
}


__global__  void kernInitParticleArrayIndicesData(int N, int *dev_particleArrayIndices) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    dev_particleArrayIndices[index] = index;
  }
}

void initParticleGridIndicesData(int N, int *particleGridIndices) {
  for (int particalIndex = 0; particalIndex < N; particalIndex++) {
    for (int x = 0; x < gridSideCount; x++) {
      for (int y = 0; y < gridSideCount; y++) {
        for (int z = 0; z < gridSideCount; z++) {
          int index = cpuGridIndex3Dto1D(x, y, z, gridSideCount);
          float cellXMininum = gridMinimum.x + x * gridCellWidth;
          float cellYMininum = gridMinimum.y + y * gridCellWidth;
          float cellZMininum = gridMinimum.z + z * gridCellWidth;
          float cellXMaximum = cellXMininum + gridCellWidth;
          float cellYMaximum = cellYMininum + gridCellWidth;
          float cellZMaximum = cellZMininum + gridCellWidth;
          if (pos[particalIndex].x >= cellXMininum && pos[particalIndex].x < cellXMaximum &&
              pos[particalIndex].y >= cellYMininum && pos[particalIndex].y < cellYMaximum &&
              pos[particalIndex].z >= cellZMininum && pos[particalIndex].z < cellZMaximum) {
            // The particle is within this cell
            particleGridIndices[particalIndex] = index;
          }
        }
      }
    }
  }
}

__global__ void kernInitParticleGridIndicesData(int N, int gridSideCount, glm::vec3 gridMinimum, float gridCellWidth, int *dev_particleGridIndices, glm::vec3 *dev_pos) {
  int particalIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (particalIndex >= N) {
    return;
  }
  // for (int x = 0; x < gridSideCount; x++) {
  //   for (int y = 0; y < gridSideCount; y++) {
  //     for (int z = 0; z < gridSideCount; z++) {
  //       int gridIndex = gridIndex3Dto1D(x, y, z, gridSideCount);
  //       if (particalIndex < N) {
  //         float cellXMininum = gridMinimum.x + x * gridCellWidth;
  //         float cellYMininum = gridMinimum.y + y * gridCellWidth;
  //         float cellZMininum = gridMinimum.z + z * gridCellWidth;
  //         float cellXMaximum = cellXMininum + gridCellWidth;
  //         float cellYMaximum = cellYMininum + gridCellWidth;
  //         float cellZMaximum = cellZMininum + gridCellWidth;
  //         if (dev_pos[particalIndex].x >= cellXMininum && dev_pos[particalIndex].x < cellXMaximum &&
  //             dev_pos[particalIndex].y >= cellYMininum && dev_pos[particalIndex].y < cellYMaximum &&
  //             dev_pos[particalIndex].z >= cellZMininum && dev_pos[particalIndex].z < cellZMaximum) {
  //           // The particle is within this cell
  //           dev_particleGridIndices[particalIndex] = gridIndex;
  //         }
  //       }
  //     }
  //   }
  // }
  glm::ivec3 gridCell = glm::floor((dev_pos[particalIndex] - gridMinimum) / gridCellWidth);
  int x = gridCell.x;
  int y = gridCell.y;
  int z = gridCell.z;
  int index = gridIndex3Dto1D(x, y, z, gridSideCount);
  if (particalIndex < N) {
    dev_particleGridIndices[particalIndex] = index;
  }
}

void initGridStartIndicesData(int N, int *particleArrayIndices, int *particleGridIndices) {
  int *dev_intKeys;
  int *dev_intValues;
  cudaMalloc((void**)&dev_intKeys, sizeof(int) * N);
  cudaMalloc((void**)&dev_intValues, sizeof(int) * N);

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, particleGridIndices, sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, particleArrayIndices, sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(particleGridIndices, dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(particleArrayIndices, dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  int i = 0, j = 1;
  gridCellStartIndices[particleGridIndices[i]] = i;
  for (; j < N;) { 
    if (particleGridIndices[j] == particleGridIndices[i]) {
      ++j;
    } else {
      gridCellEndIndices[particleGridIndices[i]] = j - 1;
      i = j;
      gridCellStartIndices[particleGridIndices[i]] = i;
      ++j;
    }
  }
  gridCellEndIndices[particleGridIndices[i]] = j - 1;

  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
}

__device__ void computeGridCellStartEndIndicesData(int index, int N, int gridCellCount, int* dev_particleGridIndices, int* dev_gridCellStartIndices, int* dev_gridCellEndIndices) {
  if (index == 0) {
    dev_gridCellStartIndices[dev_particleGridIndices[index]] = index;
  }
  if (index == N - 1) {
    dev_gridCellEndIndices[dev_particleGridIndices[index]] = index;
  }
  if (index < N - 1 && dev_particleGridIndices[index] != dev_particleGridIndices[index + 1]) {
    dev_gridCellEndIndices[dev_particleGridIndices[index]] = index;
    dev_gridCellStartIndices[dev_particleGridIndices[index + 1]] = index + 1;
  }
}

__global__ void kernInitGridIndicesData(int N, int gridCellCount, int* dev_particleGridArrayIndices, int* dev_particleGridIndices, int* dev_gridCellStartIndices, int* dev_gridCellEndIndices) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= N) {
    return;
  }
  computeGridCellStartEndIndicesData(index, N, gridCellCount, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
}

void initGridIndicesData(int N, int gridCellCount, int* dev_particleArrayIndices, int* dev_particleGridIndices, int* dev_gridCellStartIndices, int* dev_gridCellEndIndices) {
  dim3 cellBlocks((gridCellCount + blockSize - 1) / blockSize);
  kernResetIntBuffer<<<cellBlocks, blockSize>>>(gridCellCount, dev_gridCellStartIndices, -1);
  checkCUDAErrorWithLine("Reset grid cell start indices failed!");
  kernResetIntBuffer<<<cellBlocks, blockSize>>>(gridCellCount, dev_gridCellEndIndices, -1);
  checkCUDAErrorWithLine("Reset grid cell end indices failed!");

  if (N == 0) {
    return;
  }

  thrust::device_ptr<int> keys(dev_particleGridIndices);
  thrust::device_ptr<int> values(dev_particleArrayIndices);
  thrust::sort_by_key(keys, keys + N, values);
  checkCUDAErrorWithLine("Sort particle grid indices failed!");

  dim3 particleBlocks((N + blockSize - 1) / blockSize);
  kernInitGridIndicesData<<<particleBlocks, blockSize>>>(
      N, gridCellCount, dev_particleArrayIndices, dev_particleGridIndices,
      dev_gridCellStartIndices, dev_gridCellEndIndices);
  checkCUDAErrorWithLine("kernInitGridIndicesData failed!");
}

void ResetBuffers() {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  kernResetIntBuffer<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleArrayIndices, -1);
  checkCUDAErrorWithLine("Reset particle array indices failed!");
  kernResetIntBuffer<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleGridIndices, -1);
  checkCUDAErrorWithLine("Reset particle grid indices failed!");
  kernResetIntBuffer<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_gridCellStartIndices, -1);
  checkCUDAErrorWithLine("Reset grid cell start indices failed!");
  kernResetIntBuffer<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_gridCellEndIndices, -1);
  checkCUDAErrorWithLine("Reset grid cell end indices failed!");
  cudaDeviceSynchronize();
  checkCUDAErrorWithLine("CUDA device synchronize failed!");
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed

  ResetBuffers();
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  const float inverseCellWidth = 1.0f / gridCellWidth;
  // Compute indices for each particle in the grid
  kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(
      numObjects, gridSideCount, gridMinimum, inverseCellWidth,
      dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
  checkCUDAErrorWithLine("kernComputeIndices failed!");
  
  // Sort particles based on their grid indices using Thrust



}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
}

void testGpuGridInitialization() {
  const int sideCount = 3;
  const int cellCount = 27;
  const int capacity = blockSize + 1;
  thrust::device_vector<glm::vec3> testPositions(capacity);
  thrust::device_vector<int> testArrayIndices(capacity);
  thrust::device_vector<int> testGridIndices(capacity);
  thrust::device_vector<int> testStarts(cellCount);
  thrust::device_vector<int> testEnds(cellCount);
  auto devicePositions = thrust::raw_pointer_cast(testPositions.data());
  auto deviceArrayIndices = thrust::raw_pointer_cast(testArrayIndices.data());
  auto deviceGridIndices = thrust::raw_pointer_cast(testGridIndices.data());
  auto deviceStarts = thrust::raw_pointer_cast(testStarts.data());
  auto deviceEnds = thrust::raw_pointer_cast(testEnds.data());

  auto checkCuda = [](cudaError_t result, const char *operation) {
    if (result != cudaSuccess) {
      std::cerr << "[FAIL] GPU grid tests: " << operation << ": "
                << cudaGetErrorString(result) << std::endl;
      exit(EXIT_FAILURE);
    }
  };

  auto runCase = [&](const char *name, const std::vector<glm::vec3> &positions,
                     const std::vector<int> &expectedKeys) {
    auto expect = [&](const char *field, int index, int expected, int actual) {
      if (actual != expected) {
        std::cerr << "[FAIL] " << name << ": " << field << "[" << index
                  << "] expected " << expected << ", got " << actual << std::endl;
        exit(EXIT_FAILURE);
      }
    };
    int count = static_cast<int>(positions.size());
    expect("input size", 0, count, static_cast<int>(expectedKeys.size()));
    expect("capacity exceeded", 0, 0, count > capacity);
    std::vector<int> indices(count);
    std::vector<int> keys(count);
    if (count > 0) {
      checkCuda(cudaMemcpy(devicePositions, positions.data(), count * sizeof(glm::vec3),
                           cudaMemcpyHostToDevice), "upload positions");
      dim3 blocks((count + blockSize - 1) / blockSize);
      kernInitParticleArrayIndicesData<<<blocks, blockSize>>>(count, deviceArrayIndices);
      checkCuda(cudaGetLastError(), "launch array index initialization");
      kernInitParticleGridIndicesData<<<blocks, blockSize>>>(
          count, sideCount, glm::vec3(-10.0f), 10.0f, deviceGridIndices, devicePositions);
      checkCuda(cudaGetLastError(), "launch grid index initialization");
      checkCuda(cudaDeviceSynchronize(), "initialize grid indices");
      checkCuda(cudaMemcpy(indices.data(), deviceArrayIndices, count * sizeof(int),
                           cudaMemcpyDeviceToHost), "read initial array indices");
      checkCuda(cudaMemcpy(keys.data(), deviceGridIndices, count * sizeof(int),
                           cudaMemcpyDeviceToHost), "read initial grid indices");
      for (int i = 0; i < count; ++i) {
        expect("initial array index", i, i, indices[i]);
        expect("initial grid index", i, expectedKeys[i], keys[i]);
      }
    }

    initGridIndicesData(count, cellCount, deviceArrayIndices, deviceGridIndices,
                        deviceStarts, deviceEnds);
    checkCuda(cudaDeviceSynchronize(), "sort and construct cell ranges");
    if (count > 0) {
      checkCuda(cudaMemcpy(indices.data(), deviceArrayIndices, count * sizeof(int),
                           cudaMemcpyDeviceToHost), "read sorted array indices");
      checkCuda(cudaMemcpy(keys.data(), deviceGridIndices, count * sizeof(int),
                           cudaMemcpyDeviceToHost), "read sorted grid indices");
    }
    std::vector<int> expectedSorted = expectedKeys;
    std::sort(expectedSorted.begin(), expectedSorted.end());
    std::vector<int> occurrences(count, 0);
    for (int i = 0; i < count; ++i) {
      expect("sorted grid index", i, expectedSorted[i], keys[i]);
      expect("valid particle index", i, 1, indices[i] >= 0 && indices[i] < count);
      expect("key/value pairing", i, expectedKeys[indices[i]], keys[i]);
      ++occurrences[indices[i]];
    }
    for (int i = 0; i < count; ++i) {
      expect("particle occurrences", i, 1, occurrences[i]);
    }

    std::vector<int> starts(cellCount), ends(cellCount);
    std::vector<int> expectedStarts(cellCount, -1), expectedEnds(cellCount, -1);
    for (int i = 0; i < count; ++i) {
      int cell = expectedSorted[i];
      if (expectedStarts[cell] == -1) {
        expectedStarts[cell] = i;
      }
      expectedEnds[cell] = i;
    }
    checkCuda(cudaMemcpy(starts.data(), deviceStarts, cellCount * sizeof(int),
                         cudaMemcpyDeviceToHost), "read cell starts");
    checkCuda(cudaMemcpy(ends.data(), deviceEnds, cellCount * sizeof(int),
                         cudaMemcpyDeviceToHost), "read cell ends");
    for (int cell = 0; cell < cellCount; ++cell) {
      expect("cell start", cell, expectedStarts[cell], starts[cell]);
      expect("cell end", cell, expectedEnds[cell], ends[cell]);
    }
    std::cout << "[PASS] " << name << std::endl;
  };

  runCase("Mixed cells and exact boundaries",
          {{5, 5, 5}, {-5, -5, -5}, {15, 15, 15},
           {1, 2, 3}, {0, -5, -5}, {-10, -10, -10},
           {-0.001f, -5, -5}, {10, -5, -5}, {-5, 0, -5}, {-5, -5, 0}},
          {13, 0, 26, 13, 1, 0, 0, 2, 3, 9});
  runCase("Single particle clears previous cells", {{5, 5, 5}}, {13});
  runCase("All particles in one cell", std::vector<glm::vec3>(6, glm::vec3(-5)),
          std::vector<int>(6, 0));

  std::vector<glm::vec3> positions(capacity, glm::vec3(5));
  std::vector<int> keys(capacity, 13);
  runCase("One cell spans CUDA blocks", positions, keys);
  positions.back() = glm::vec3(15);
  keys.back() = 26;
  runCase("Cell transition at CUDA block boundary", positions, keys);
  runCase("Empty rebuild clears all cells", {}, {});
}

void Boids::unitTest() {
  testGpuGridInitialization();
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  // int *dev_intKeys;
  // int *dev_intValues;
  // int N = 10;

  // std::unique_ptr<int[]>intKeys{ new int[N] };
  // std::unique_ptr<int[]>intValues{ new int[N] };

  // intKeys[0] = 0; intValues[0] = 0;
  // intKeys[1] = 1; intValues[1] = 1;
  // intKeys[2] = 0; intValues[2] = 2;
  // intKeys[3] = 3; intValues[3] = 3;
  // intKeys[4] = 0; intValues[4] = 4;
  // intKeys[5] = 2; intValues[5] = 5;
  // intKeys[6] = 2; intValues[6] = 6;
  // intKeys[7] = 0; intValues[7] = 7;
  // intKeys[8] = 5; intValues[8] = 8;
  // intKeys[9] = 6; intValues[9] = 9;

  // cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  // checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  // cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  // checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  // dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // std::cout << "before unstable sort: " << std::endl;
  // for (int i = 0; i < N; i++) {
  //   std::cout << "  key: " << intKeys[i];
  //   std::cout << " value: " << intValues[i] << std::endl;
  // }

  // // How to copy data to the GPU
  // cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  // cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // // Wrap device vectors in thrust iterators for use with thrust.
  // thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  // thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // // LOOK-2.1 Example for using thrust::sort_by_key
  // thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // // How to copy data back to the CPU side from the GPU
  // cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  // cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  // checkCUDAErrorWithLine("memcpy back failed!");

  // std::cout << "after unstable sort: " << std::endl;
  // for (int i = 0; i < N; i++) {
  //   std::cout << "  key: " << intKeys[i];
  //   std::cout << " value: " << intValues[i] << std::endl;
  // }

  // // cleanup
  // cudaFree(dev_intKeys);
  // cudaFree(dev_intValues);
  // checkCUDAErrorWithLine("cudaFree failed!");

  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  // cpu init version
  // kernInitParticleArrayIndicesData<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleArrayIndices);
  // checkCUDAErrorWithLine("kernInitParticleArrayIndicesData failed!");

  // cudaMemcpy(particleArrayIndices.get(), dev_particleArrayIndices, numObjects * sizeof(int), cudaMemcpyDeviceToHost);
  // checkCUDAErrorWithLine("cudaMemcpy particleArrayIndices failed!");


  // initParticleGridIndicesData(numObjects, particleGridIndices.get());

  // std::cout << "Before sorting:" << std::endl;
  // for(int i = 0; i < numObjects; i++) {
  //   std::cout << "Particle " << i << ": Grid Index = " << particleGridIndices[i] << ", Array Index = " << particleArrayIndices[i] << " Pos = " << pos[i].x << ", " << pos[i].y << ", " << pos[i].z << std::endl;
  // }

  // initGridStartIndicesData(numObjects, particleArrayIndices.get(), particleGridIndices.get());
  // std::cout << "After sorting:" << std::endl;
  // for(int i = 0; i < numObjects; i++) {
  //   std::cout << "Particle " << i << ": Grid Index = " << particleGridIndices[i] << ", Array Index = " << particleArrayIndices[i] << std::endl;
  // }
  // for (int i = 0; i < gridCellCount; i++) {
  //   if (gridCellStartIndices[i] == -1 && gridCellEndIndices[i] == -1) {
  //     continue;
  //   }
  //   std::cout << "Grid cell " << i << " starts at " << gridCellStartIndices[i] << " and ends at " << gridCellEndIndices[i] << std::endl;
  // }

  // gpu init version
  kernInitParticleArrayIndicesData<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleArrayIndices);
  checkCUDAErrorWithLine("kernInitParticleArrayIndicesData failed!");
  kernInitParticleGridIndicesData<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount, gridMinimum, gridCellWidth, dev_particleGridIndices, dev_pos);
  checkCUDAErrorWithLine("kernInitParticleGridIndicesData failed!");
  std::unique_ptr<glm::vec3[]> cpu_pos = std::make_unique<glm::vec3[]>(numObjects);
  std::unique_ptr<int[]> cpu_particleArrayIndices = std::make_unique<int[]>(numObjects);
  std::unique_ptr<int[]> cpu_particleGridIndices = std::make_unique<int[]>(numObjects);
  std::unique_ptr<int[]> cpu_gridCellStartIndices = std::make_unique<int[]>(gridCellCount);
  std::unique_ptr<int[]> cpu_gridCellEndIndices = std::make_unique<int[]>(gridCellCount);

  cudaMemcpy(cpu_pos.get(), dev_pos, numObjects * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("cudaMemcpy GPU test positions failed!");

  auto printGpuIndices = [&](const char *label) {
    cudaMemcpy(cpu_particleArrayIndices.get(), dev_particleArrayIndices, numObjects * sizeof(int), cudaMemcpyDeviceToHost);
    checkCUDAErrorWithLine("cudaMemcpy GPU test particle array indices failed!");
    cudaMemcpy(cpu_particleGridIndices.get(), dev_particleGridIndices, numObjects * sizeof(int), cudaMemcpyDeviceToHost);
    checkCUDAErrorWithLine("cudaMemcpy GPU test particle grid indices failed!");

    std::cout << label << std::endl;
    for (int i = 0; i < numObjects; i++) {
      int particleIndex = cpu_particleArrayIndices[i];
      const glm::vec3 &position = cpu_pos[particleIndex];
      std::cout << "Slot " << i << ": Grid Index = " << cpu_particleGridIndices[i]
                << ", Array Index = " << particleIndex
                << ", Pos = " << position.x << ", " << position.y << ", " << position.z << std::endl;
    }
  };

  printGpuIndices("GPU before unstable sort:");
  initGridIndicesData(numObjects, gridCellCount, dev_particleArrayIndices, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
  printGpuIndices("GPU after unstable sort:");

  cudaMemcpy(cpu_gridCellStartIndices.get(), dev_gridCellStartIndices, gridCellCount * sizeof(int), cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("cudaMemcpy GPU test grid cell start indices failed!");
  cudaMemcpy(cpu_gridCellEndIndices.get(), dev_gridCellEndIndices, gridCellCount * sizeof(int), cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("cudaMemcpy GPU test grid cell end indices failed!");

  std::cout << "GPU grid cell ranges (inclusive):" << std::endl;
  for (int i = 0; i < gridCellCount; i++) {
    if (cpu_gridCellStartIndices[i] == -1) {
      continue;
    }
    std::cout << "Grid cell " << i << " starts at " << cpu_gridCellStartIndices[i]
              << " and ends at " << cpu_gridCellEndIndices[i] << std::endl;
  }

  return;
}
