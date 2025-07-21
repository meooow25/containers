module Utils.SetMerge
  ( WhenMissingSpec(..)
  ) where
  
import Test.QuickCheck (Arbitrary(..), CoArbitrary, Function, Fun, oneof)

data WhenMissingSpec a
  = DropMissingSpec
  | PreserveMissingSpec
  | FilterMissingSpec (Fun a Bool)
  deriving Show

instance (Arbitrary a, CoArbitrary a, Function a)
  => Arbitrary (WhenMissingSpec a) where
  arbitrary = oneof
    [ pure DropMissingSpec
    , pure PreserveMissingSpec
    , FilterMissingSpec <$> arbitrary
    ]
  shrink s = case s of
    DropMissingSpec -> []
    PreserveMissingSpec -> []
    FilterMissingSpec f -> FilterMissingSpec <$> shrink f

