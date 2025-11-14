/**
 * useCapsule Hook
 *
 * Custom React hook for managing game capsules in the CapsuleOS system.
 * Provides functionality for creating, retrieving, updating, and managing
 * game capsules with GΛLYPH expressions.
 */

import { useState, useCallback, useEffect, useRef } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'react-hot-toast'

import { glyphParserWasm, type WasmExpression, type GameComponents } from '@/lib/glyph-parser-wasm'

// ============================================================================
// Type Definitions
// ============================================================================

export interface CapsuleMetadata {
  id: string
  version: string
  created_at: number
  parent_cid?: string
  genesis_cid: string
  game_cid: string
  user_id?: string
  tags: string[]
}

export interface GameManifest {
  id: string
  title: string
  story: string
  rules: Record<string, any>
  code: string
  balance: number
  genre?: string
  theme?: string
  created_at: number
  llm_outputs: LLMOutput[]
}

export interface LLMOutput {
  llm_name: string
  llm_role: 'narrative' | 'mechanics' | 'assets' | 'balance'
  glyph_expression: string
  processed_at: number
  confidence?: number
}

export interface GameCapsule {
  metadata: CapsuleMetadata
  manifest: GameManifest
  expression?: WasmExpression
  hash?: string
}

export interface CapsuleCreationRequest {
  title: string
  description: string
  genre?: string
  complexity: 'simple' | 'medium' | 'complex'
  theme?: string
  components: {
    narrative?: string
    mechanics?: string
    assets?: string
    balance?: string
  }
}

export interface CapsuleUpdateRequest {
  title?: string
  description?: string
  components?: Partial<{
    narrative: string
    mechanics: string
    assets: string
    balance: string
  }>
}

export interface EvolutionRequest {
  parent_cid: string
  evolution_instructions: string
  context?: Record<string, any>
}

// ============================================================================
// API Functions
// ============================================================================

class CapsuleAPI {
  private baseUrl: string

  constructor(baseUrl: string = '/api/v1') {
    this.baseUrl = baseUrl
  }

  async createCapsule(request: CapsuleCreationRequest): Promise<GameCapsule> {
    const response = await fetch(`${this.baseUrl}/games/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user_id: 'demo-user', // TODO: Get from auth context
        input_text: request.description,
        context: {
          title: request.title,
          genre: request.genre,
          complexity: request.complexity,
          theme: request.theme,
          components: request.components,
        },
      }),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.message || 'Failed to create capsule')
    }

    const data = await response.json()
    return this.transformResponseToCapsule(data)
  }

  async getCapsule(cid: string): Promise<GameCapsule | null> {
    const response = await fetch(`${this.baseUrl}/games/${cid}`)

    if (response.status === 404) {
      return null
    }

    if (!response.ok) {
      throw new Error('Failed to retrieve capsule')
    }

    const data = await response.json()
    return this.transformResponseToCapsule(data)
  }

  async updateCapsule(cid: string, request: CapsuleUpdateRequest): Promise<GameCapsule> {
    // Note: This would be implemented as an evolution in CapsuleOS
    const evolutionRequest: EvolutionRequest = {
      parent_cid: cid,
      evolution_instructions: request.description || 'Update capsule',
      context: {
        title: request.title,
        components: request.components,
      },
    }

    return this.evolveCapsule(evolutionRequest)
  }

  async evolveCapsule(request: EvolutionRequest): Promise<GameCapsule> {
    const response = await fetch(`${this.baseUrl}/games/${request.parent_cid}/evolve`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        parent_cid: request.parent_cid,
        user_id: 'demo-user', // TODO: Get from auth context
        evolution_instructions: request.evolution_instructions,
        context: request.context,
      }),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.message || 'Failed to evolve capsule')
    }

    const data = await response.json()
    return this.transformResponseToCapsule(data)
  }

  async getLineage(cid: string): Promise<string[]> {
    const response = await fetch(`${this.baseUrl}/games/${cid}/lineage`)

    if (!response.ok) {
      throw new Error('Failed to get capsule lineage')
    }

    const data = await response.json()
    return data.lineage || []
  }

  async getUserCapsules(userId: string): Promise<GameCapsule[]> {
    // This would need to be implemented in the API
    // For now, return empty array
    return []
  }

  private transformResponseToCapsule(data: any): GameCapsule {
    return {
      metadata: {
        id: data.id || data.game_cid,
        version: '1.0.0',
        created_at: data.created_at || Date.now(),
        parent_cid: data.parent_cid,
        genesis_cid: data.genesis_cid || 'genesis-root',
        game_cid: data.game_cid,
        user_id: data.user_id,
        tags: data.tags || ['game', 'generated'],
      },
      manifest: {
        id: data.id,
        title: data.title,
        story: data.story,
        rules: data.rules || {},
        code: data.code || '',
        balance: data.balance || 0.5,
        genre: data.genre,
        theme: data.theme,
        created_at: data.created_at || Date.now(),
        llm_outputs: data.llm_outputs || [],
      },
    }
  }
}

// ============================================================================
// Custom Hook Implementation
// ============================================================================

const capsuleAPI = new CapsuleAPI()

export function useCapsule(cid?: string) {
  const queryClient = useQueryClient()
  const [isParsing, setIsParsing] = useState(false)

  // Query for single capsule
  const {
    data: capsule,
    isLoading: isLoadingCapsule,
    error: capsuleError,
    refetch: refetchCapsule,
  } = useQuery({
    queryKey: ['capsule', cid],
    queryFn: () => (cid ? capsuleAPI.getCapsule(cid) : Promise.resolve(null)),
    enabled: !!cid,
    staleTime: 5 * 60 * 1000, // 5 minutes
  })

  // Query for capsule lineage
  const {
    data: lineage,
    isLoading: isLoadingLineage,
    refetch: refetchLineage,
  } = useQuery({
    queryKey: ['capsule-lineage', cid],
    queryFn: () => (cid ? capsuleAPI.getLineage(cid) : Promise.resolve([])),
    enabled: !!cid,
    staleTime: 10 * 60 * 1000, // 10 minutes
  })

  // Mutation for creating capsules
  const createCapsuleMutation = useMutation({
    mutationFn: capsuleAPI.createCapsule,
    onSuccess: (newCapsule) => {
      queryClient.invalidateQueries({ queryKey: ['capsules'] })
      queryClient.setQueryData(['capsule', newCapsule.metadata.game_cid], newCapsule)
      toast.success(`Game "${newCapsule.manifest.title}" created successfully!`)
    },
    onError: (error: Error) => {
      toast.error(`Failed to create game: ${error.message}`)
    },
  })

  // Mutation for updating capsules
  const updateCapsuleMutation = useMutation({
    mutationFn: ({ cid, request }: { cid: string; request: CapsuleUpdateRequest }) =>
      capsuleAPI.updateCapsule(cid, request),
    onSuccess: (updatedCapsule) => {
      queryClient.setQueryData(['capsule', updatedCapsule.metadata.game_cid], updatedCapsule)
      queryClient.invalidateQueries({ queryKey: ['capsule-lineage', updatedCapsule.metadata.game_cid] })
      toast.success(`Game "${updatedCapsule.manifest.title}" updated successfully!`)
    },
    onError: (error: Error) => {
      toast.error(`Failed to update game: ${error.message}`)
    },
  })

  // Mutation for evolving capsules
  const evolveCapsuleMutation = useMutation({
    mutationFn: capsuleAPI.evolveCapsule,
    onSuccess: (evolvedCapsule) => {
      queryClient.invalidateQueries({ queryKey: ['capsules'] })
      queryClient.setQueryData(['capsule', evolvedCapsule.metadata.game_cid], evolvedCapsule)
      toast.success(`Game evolved to version: ${evolvedCapsule.metadata.game_cid}`)
    },
    onError: (error: Error) => {
      toast.error(`Failed to evolve game: ${error.message}`)
    },
  })

  // ============================================================================
  // Helper Functions
  // ============================================================================

  const createCapsule = useCallback(async (request: CapsuleCreationRequest) => {
    return createCapsuleMutation.mutateAsync(request)
  }, [createCapsuleMutation])

  const updateCapsule = useCallback(async (cid: string, request: CapsuleUpdateRequest) => {
    return updateCapsuleMutation.mutateAsync({ cid, request })
  }, [updateCapsuleMutation])

  const evolveCapsule = useCallback(async (request: EvolutionRequest) => {
    return evolveCapsuleMutation.mutateAsync(request)
  }, [evolveCapsuleMutation])

  const parseGameExpression = useCallback(async (expression: string): Promise<{
    isValid: boolean
    components: GameComponents
    hash?: string
    errors: string[]
  }> => {
    setIsParsing(true)
    try {
      await glyphParserWasm.initialize()

      const validation = await glyphParserWasm.validateExpression(expression)

      if (!validation.is_valid) {
        return {
          isValid: false,
          components: { is_game_lambda: false },
          errors: validation.errors,
        }
      }

      const parseResult = await glyphParserWasm.parseExpression(expression)

      return {
        isValid: true,
        components: parseResult.components,
        hash: parseResult.hash,
        errors: [],
      }
    } catch (error) {
      return {
        isValid: false,
        components: { is_game_lambda: false },
        errors: [error instanceof Error ? error.message : 'Unknown parsing error'],
      }
    } finally {
      setIsParsing(false)
    }
  }, [])

  const buildGameExpression = useCallback((components: {
    narrative?: string
    mechanics?: string
    assets?: string
    balance?: string
  }): string => {
    const parts: string[] = []

    if (components.narrative) {
      parts.push(`λnarrative.story="${components.narrative}"`)
    }
    if (components.mechanics) {
      parts.push(`λmechanics.${components.mechanics}`)
    }
    if (components.assets) {
      parts.push(`λassets.${components.assets}`)
    }
    if (components.balance) {
      parts.push(`λbalance.${components.balance}`)
    }

    if (parts.length === 0) {
      return 'λgame.()'
    }

    return `λgame.(${parts.join(' ')})`
  }, [])

  const validateCapsuleIntegrity = useCallback(async (capsule: GameCapsule): Promise<{
    isValid: boolean
    issues: string[]
  }> => {
    const issues: string[] = []

    // Check if capsule has required fields
    if (!capsule.manifest.title) {
      issues.push('Missing game title')
    }
    if (!capsule.manifest.story) {
      issues.push('Missing game story')
    }
    if (!capsule.manifest.code) {
      issues.push('Missing game code')
    }

    // Check balance range
    if (capsule.manifest.balance < 0 || capsule.manifest.balance > 1) {
      issues.push('Balance score must be between 0 and 1')
    }

    // Check if we have LLM outputs
    if (capsule.manifest.llm_outputs.length !== 4) {
      issues.push('Expected exactly 4 LLM outputs')
    }

    // Parse and validate GΛLYPH expressions if present
    if (capsule.manifest.llm_outputs.length > 0) {
      for (const output of capsule.manifest.llm_outputs) {
        const parseResult = await parseGameExpression(output.glyph_expression)
        if (!parseResult.isValid) {
          issues.push(`Invalid GΛLYPH expression from ${output.llm_role}: ${parseResult.errors.join(', ')}`)
        }
      }
    }

    return {
      isValid: issues.length === 0,
      issues,
    }
  }, [parseGameExpression])

  const refreshCapsule = useCallback(() => {
    if (cid) {
      refetchCapsule()
      refetchLineage()
    }
  }, [cid, refetchCapsule, refetchLineage])

  // ============================================================================
  // Return State and Actions
  // ============================================================================

  return {
    // State
    capsule,
    lineage,
    isLoading: isLoadingCapsule || isLoadingLineage,
    error: capsuleError,
    isParsing,

    // Mutations
    createCapsule,
    updateCapsule,
    evolveCapsule,

    // Actions
    refreshCapsule,
    parseGameExpression,
    buildGameExpression,
    validateCapsuleIntegrity,

    // Mutation states
    isCreating: createCapsuleMutation.isPending,
    isUpdating: updateCapsuleMutation.isPending,
    isEvolving: evolveCapsuleMutation.isPending,
  }
}

// ============================================================================
// Additional Hooks for Specific Use Cases
// ============================================================================

export function useCapsulesList(userId?: string) {
  return useQuery({
    queryKey: ['capsules', userId],
    queryFn: () => capsuleAPI.getUserCapsules(userId || 'demo-user'),
    staleTime: 2 * 60 * 1000, // 2 minutes
  })
}

export function useCapsuleEvolution(parentCid: string) {
  const { mutate: evolve, isPending } = useMutation({
    mutationFn: (instructions: string) => capsuleAPI.evolveCapsule({
      parent_cid: parentCid,
      evolution_instructions: instructions,
    }),
    onSuccess: (evolvedCapsule) => {
      toast.success(`Game evolved successfully! New CID: ${evolvedCapsule.metadata.game_cid}`)
    },
    onError: (error: Error) => {
      toast.error(`Evolution failed: ${error.message}`)
    },
  })

  return {
    evolve,
    isEvolving: isPending,
  }
}

export function useCapsuleValidation() {
  const [isValidating, setIsValidating] = useState(false)

  const validateExpression = useCallback(async (expression: string): Promise<{
    isValid: boolean
    errors: string[]
    warnings: string[]
  }> => {
    setIsValidating(true)
    try {
      await glyphParserWasm.initialize()
      const result = await glyphParserWasm.validateExpression(expression)
      return {
        isValid: result.is_valid,
        errors: result.errors,
        warnings: result.warnings,
      }
    } catch (error) {
      return {
        isValid: false,
        errors: [error instanceof Error ? error.message : 'Validation error'],
        warnings: [],
      }
    } finally {
      setIsValidating(false)
    }
  }, [])

  return {
    validateExpression,
    isValidating,
  }
}