#pragma once

namespace flash_attn {

inline constexpr int kHeadDim = 64;
inline constexpr int kQueryTile = 64;
inline constexpr int kKeyTile = 64;
inline constexpr int kPaddedHeadDim = 68;

}  // namespace flash_attn
