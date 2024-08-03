/****************************************************************************
 * Copyright (c) 2022-2023 by Oak Ridge National Laboratory                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of CabanaPD. CabanaPD is distributed under a           *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CabanaDEMInput_HPP
#define CabanaDEMInput_HPP

#include <fstream>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

namespace CabanaDEM
{
  class Inputs
  {
  public:
    Inputs( const std::string filename )
    {
      // Get user inputs.
      inputs = parse( filename );

      // Number of steps.
      double tf = inputs["final_time"]["value"];
      double dt = inputs["timestep"]["value"];

      int num_steps = tf / dt;
      inputs["num_steps"]["value"] = num_steps;
    }
    ~Inputs() {}

    // Parse JSON file.
    inline nlohmann::json parse( const std::string& filename )
    {
        std::ifstream stream( filename );
        return nlohmann::json::parse( stream );
    }

    // Get a single input.
    auto operator[]( std::string label ) { return inputs[label]["value"]; }

    // Get a single input.
    std::string units( std::string label )
    {
        if ( inputs[label].contains( "units" ) )
            return inputs[label]["units"];
        else
            return "";
    }

    // Check a key exists.
    bool contains( std::string label ) { return inputs.contains( label ); }

  protected:
    nlohmann::json inputs;
};

} // namespace CabanaDEM

#endif
